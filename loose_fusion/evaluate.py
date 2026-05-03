"""
Evaluate the loose-fusion pipeline.

Pipeline per sequence:

    1. Run AirIMU once over the full IMU stream to obtain corrected IMU
       and per-sample uncertainty.
    2. Run AirIO once (with ground-truth or EKF attitude) to obtain
       per-sample body-frame velocity + uncertainty.
    3. Run BranchA once over consecutive frame pairs to obtain a
       relative ``(ΔR, Δt)`` per pair.
    4. Step the FusionEKF sample-by-sample:
         - predict from corrected IMU,
         - update with AirIO velocity at every IMU sample,
         - update with vision velocity + rotation at every frame.
    5. Compute ATE / RTE / mean rotation error and write plots.

This script does *not* train anything — it only evaluates. The three
backbones must be pre-trained:

    --vision_checkpoint    path to vision_only/checkpoints/best.pt
    --airio_checkpoint     path to imu_only/checkpoints/airio/best.pt
    --airimu_checkpoint    optional path to imu_only/checkpoints/airimu/best.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

# Sibling-branch imports.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from fusion_common.dataset import _PairedSequence  # noqa: E402
from fusion_common.metrics import trajectory_metrics  # noqa: E402
from fusion_ekf import FusionEKF  # noqa: E402
from imu_only.dataset import IMUWindowDataset  # noqa: E402
from imu_only.model import AirIMUNet, AirIONet  # noqa: E402
from vision_only.model import BranchA  # noqa: E402

import torchvision.transforms.functional as TF  # noqa: E402
from PIL import Image  # noqa: E402

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Loose-fusion evaluation.")
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--vision_checkpoint", type=str, required=True)
    p.add_argument("--airio_checkpoint", type=str, required=True)
    p.add_argument("--airimu_checkpoint", type=str, default=None)
    p.add_argument("--results_dir", type=str, default="results/loose_fusion")
    p.add_argument("--imu_rate", type=float, default=1000.0)
    p.add_argument("--img_height", type=int, default=224)
    p.add_argument("--img_width", type=int, default=224)
    p.add_argument("--vision_chunk", type=int, default=10)
    p.add_argument("--airio_chunk", type=int, default=1000)
    p.add_argument("--rte_interval", type=float, default=5.0)
    p.add_argument("--vision_vel_sigma", type=float, default=0.05)
    p.add_argument("--vision_rot_sigma_deg", type=float, default=0.5)
    p.add_argument("--no_vision", action="store_true",
                   help="Disable the vision update — runs IMU-only baseline.")
    p.add_argument("--no_imu_update", action="store_true",
                   help="Disable the AirIO velocity update.")
    p.add_argument("--sequences", type=str, nargs="*", default=None)
    return p.parse_args()


# ---------------------------------------------------------------------------
def _load_image(path: str, h: int, w: int) -> torch.Tensor:
    img = Image.open(path).convert("RGB").resize((w, h), Image.BILINEAR)
    tensor = TF.to_tensor(img)
    return TF.normalize(tensor, _IMAGENET_MEAN, _IMAGENET_STD)


def _vision_predict_sequence(
    branch_a: BranchA,
    seq: _PairedSequence,
    device: torch.device,
    chunk: int,
    img_h: int,
    img_w: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Run BranchA across the whole sequence and return per-pair (R, t)."""
    n_pairs = seq.n_cam - 1
    rel_R = np.zeros((n_pairs, 3, 3), dtype=np.float64)
    rel_t = np.zeros((n_pairs, 3), dtype=np.float64)
    hidden = None
    pos = 0
    while pos < n_pairs:
        end = min(pos + chunk, n_pairs)
        T = end - pos
        f0 = torch.empty((1, T, 3, img_h, img_w), dtype=torch.float32)
        f1 = torch.empty_like(f0)
        for i in range(T):
            f0[0, i] = _load_image(seq.frame_files[pos + i], img_h, img_w)
            f1[0, i] = _load_image(seq.frame_files[pos + i + 1], img_h, img_w)
        f0 = f0.to(device)
        f1 = f1.to(device)
        trans, _, R_pred, hidden, _ = branch_a(f0, f1, hidden=hidden)
        hidden = (hidden[0].detach(), hidden[1].detach())
        rel_R[pos:end] = R_pred[0].detach().cpu().numpy()
        rel_t[pos:end] = trans[0].detach().cpu().numpy()
        pos = end
    return rel_R, rel_t


def _airio_predict_sequence(
    airimu: AirIMUNet | None,
    airio: AirIONet,
    seq: _PairedSequence,
    device: torch.device,
    chunk: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run AirIMU + AirIO across the IMU stream, returning per-sample arrays."""
    n = seq.n
    v_pred = np.zeros((n, 3), dtype=np.float64)
    log_var = np.zeros((n, 3), dtype=np.float64)
    imu_log_var = np.zeros((n, 6), dtype=np.float64)
    acc_corr = seq.acc.astype(np.float64).copy()
    gyro_corr = seq.gyro.astype(np.float64).copy()

    acc_t = torch.from_numpy(seq.acc).to(device)
    gyro_t = torch.from_numpy(seq.gyro).to(device)
    att_t = torch.from_numpy(seq.attitude).to(device).float()

    if airimu is not None:
        with torch.no_grad():
            a_hat, g_hat, lv_imu = airimu.correct(
                acc_t.unsqueeze(0), gyro_t.unsqueeze(0)
            )
        acc_corr = a_hat[0].cpu().numpy().astype(np.float64)
        gyro_corr = g_hat[0].cpu().numpy().astype(np.float64)
        imu_log_var = lv_imu[0].cpu().numpy().astype(np.float64)
        # Re-build the torch tensors of corrected IMU for AirIO.
        acc_in = a_hat
        gyro_in = g_hat
    else:
        acc_in = acc_t.unsqueeze(0)
        gyro_in = gyro_t.unsqueeze(0)

    pos = 0
    while pos < n:
        end = min(pos + chunk, n)
        a = acc_in[:, pos:end]
        g = gyro_in[:, pos:end]
        att = att_t.unsqueeze(0)[:, pos:end]
        v, lv = airio(a, g, att)
        v_pred[pos:end] = v[0].detach().cpu().numpy()
        log_var[pos:end] = lv[0].detach().cpu().numpy()
        pos = end

    return v_pred, log_var, imu_log_var, acc_corr, gyro_corr


# ---------------------------------------------------------------------------
def _run_pipeline(
    seq: _PairedSequence,
    branch_a: BranchA,
    airimu: AirIMUNet | None,
    airio: AirIONet,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, np.ndarray]:
    """Runs all three networks then steps the FusionEKF."""
    rel_R_vis, rel_t_vis = _vision_predict_sequence(
        branch_a, seq, device, args.vision_chunk, args.img_height, args.img_width
    )
    v_pred, log_var, imu_log_var, acc_corr, gyro_corr = _airio_predict_sequence(
        airimu, airio, seq, device, args.airio_chunk
    )

    n = seq.n          # number of IMU samples
    dt = seq.dt        # IMU timestep (e.g. 1/1000 s)
    dt_frame = seq.dt_cam  # camera frame interval (e.g. 1/100 s)

    # Build mapping: IMU sample index → camera frame index k (pair k-1 → k).
    imu_to_frame: dict[int, int] = {
        int(idx): k for k, idx in enumerate(seq.frame_imu_indices)
    }

    ekf = FusionEKF()
    ekf.reset(
        R=seq.R[0].copy(),
        v=seq.R[0] @ seq.v_body[0],
        p=seq.p[0].copy(),
    )

    pos_traj = np.zeros((n, 3), dtype=np.float64)
    R_traj = np.zeros((n, 3, 3), dtype=np.float64)
    pos_traj[0] = ekf.p
    R_traj[0] = ekf.R

    for i in range(1, n):
        ekf.predict(
            acc=acc_corr[i - 1],
            gyro=gyro_corr[i - 1],
            dt=dt,
            imu_log_var=imu_log_var[i - 1] if airimu is not None else None,
        )
        if not args.no_imu_update:
            ekf.update_velocity(v_body_meas=v_pred[i], log_var=log_var[i])

        # Vision update only at camera frame times.
        if not args.no_vision and i in imu_to_frame:
            k = imu_to_frame[i]
            if k > 0:  # frame pair (k-1) → k exists in rel_R_vis / rel_t_vis
                ekf.update_vision_velocity(
                    delta_t_vis=rel_t_vis[k - 1],
                    dt_frame=dt_frame,
                    sigma=args.vision_vel_sigma,
                )
                ekf.update_vision_rotation(
                    delta_R_vis=rel_R_vis[k - 1],
                    sigma_deg=args.vision_rot_sigma_deg,
                )

        pos_traj[i] = ekf.p
        R_traj[i] = ekf.R

    return {
        "pos_pred": pos_traj,
        "R_pred": R_traj,
        "pos_gt": seq.p,
        "R_gt": seq.R,
        "rel_R_vis": rel_R_vis,
        "rel_t_vis": rel_t_vis,
    }


# ---------------------------------------------------------------------------
def _plot(name: str, out: dict, metrics: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    pos_pred = out["pos_pred"]
    pos_gt = out["pos_gt"]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(pos_gt[:, 0], pos_gt[:, 1], "b-", label="ground truth")
    ax.plot(pos_pred[:, 0], pos_pred[:, 1], "r-", label="loose fusion")
    ax.scatter([pos_gt[0, 0]], [pos_gt[0, 1]], c="green", s=60, label="start", zorder=5)
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_title(f"{name} — top-down trajectory (loose fusion)")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"{name}_trajectory.png", dpi=150)
    plt.close(fig)

    for kind, series, ylabel in [
        ("trans", metrics["per_frame_trans"], "Position error [m]"),
        ("rot", metrics["per_frame_rot_deg"], "Rotation error [deg]"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(series, "r-")
        ax.set_xlabel("Sample index")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{name} — per-sample {kind} error")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / f"{name}_{kind}_error.png", dpi=150)
        plt.close(fig)


# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    branch_a = BranchA().to(device).eval()
    state = torch.load(args.vision_checkpoint, map_location=device)
    branch_a.load_state_dict(state.get("model", state))

    airio = AirIONet().to(device).eval()
    state = torch.load(args.airio_checkpoint, map_location=device)
    airio.load_state_dict(state.get("model", state))

    airimu = None
    if args.airimu_checkpoint:
        airimu = AirIMUNet().to(device).eval()
        st = torch.load(args.airimu_checkpoint, map_location=device)
        airimu.load_state_dict(st.get("model", st))

    seq_names = args.sequences or IMUWindowDataset.list_sequences(args.data_root)
    out_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rte_step = max(1, int(round(args.imu_rate * args.rte_interval)))

    print(f"{'sequence':<20} {'ATE [m]':>10} {'RTE [m]':>10} {'rot [deg]':>12} {'samples':>8}")
    print("-" * 64)
    aggregate = {"ate": [], "rte": [], "rot_deg": []}
    with torch.no_grad():
        for name in seq_names:
            seq = _PairedSequence(Path(args.data_root) / name, imu_rate=args.imu_rate)
            out = _run_pipeline(seq, branch_a, airimu, airio, device, args)
            metrics = trajectory_metrics(
                pos_pred=out["pos_pred"],
                pos_gt=out["pos_gt"],
                R_pred=out["R_pred"],
                R_gt=out["R_gt"],
                rte_step=rte_step,
                aligned=False,
            )
            print(f"{name:<20} {metrics['ate']:>10.4f} {metrics['rte']:>10.4f} "
                  f"{metrics['rot_deg']:>12.4f} {seq.n:>8d}")
            _plot(name, out, metrics, out_dir)
            aggregate["ate"].append(metrics["ate"])
            aggregate["rte"].append(metrics["rte"])
            aggregate["rot_deg"].append(metrics["rot_deg"])

    print("-" * 64)
    print(f"{'mean':<20} {np.mean(aggregate['ate']):>10.4f} "
          f"{np.mean(aggregate['rte']):>10.4f} "
          f"{np.mean(aggregate['rot_deg']):>12.4f}")
    print(f"\nPlots written to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
