"""
Evaluate the full IMU-only pipeline (AirIMU + AirIO + EKF).

For each test sequence the script:

    1. Loads the raw IMU stream and ground-truth pose stream.
    2. Runs AirIMU (if a checkpoint is provided) to obtain the corrected
       IMU and per-frame uncertainty.
    3. Runs AirIO to obtain a per-sample body-frame velocity prediction
       and its diagonal uncertainty. Two attitude-source modes are
       supported:
         - ``--attitude_source gt``  uses the ground-truth attitude
           (matches the paper's training-time protocol; useful as an
           upper bound on AirIO accuracy).
         - ``--attitude_source ekf`` re-feeds the EKF-estimated attitude
           into AirIO every ``--airio_chunk`` samples, mimicking real-time
           inference where ground-truth orientation is unavailable.
    4. Steps the velocity-measurement EKF sample by sample, recording the
       integrated trajectory.
    5. Reports ATE (Eq. 13) and RTE (Eq. 14) plus the mean rotation error
       in degrees, and writes trajectory + per-frame error plots.

Usage::

    python evaluate.py \\
        --data_root /path/to/dataset \\
        --airio_checkpoint checkpoints/airio/best.pt \\
        --airimu_checkpoint checkpoints/airimu/best.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from dataset import IMUWindowDataset, _Sequence
from ekf import VelocityEKF
from model import AirIMUNet, AirIONet
from utils import (
    GRAVITY,
    attitude_logmap_from_poses,
    so3_log_np,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate AirIMU + AirIO + EKF.")
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--airio_checkpoint", type=str, required=True)
    p.add_argument("--airimu_checkpoint", type=str, default=None)
    p.add_argument("--results_dir", type=str, default="results/imu_only")
    p.add_argument("--imu_rate", type=float, default=1000.0)
    p.add_argument("--airio_chunk", type=int, default=1000,
                   help="Number of IMU samples per AirIO forward pass.")
    p.add_argument("--rte_interval", type=float, default=5.0,
                   help="RTE evaluation interval in seconds (paper: 5 s).")
    p.add_argument("--attitude_source", type=str, default="gt",
                   choices=["gt", "ekf"])
    p.add_argument("--sequences", type=str, nargs="*", default=None)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------
def _airio_forward_chunked(
    airio: AirIONet,
    airimu: AirIMUNet | None,
    acc: torch.Tensor,
    gyro: torch.Tensor,
    attitude: torch.Tensor,
    chunk: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run AirIMU+AirIO over a long sequence in non-overlapping chunks."""
    n = acc.shape[0]
    v_pred = np.zeros((n, 3), dtype=np.float64)
    log_var = np.zeros((n, 3), dtype=np.float64)
    imu_log_var = np.zeros((n, 6), dtype=np.float64)

    pos = 0
    while pos < n:
        end = min(pos + chunk, n)
        a = acc[pos:end].unsqueeze(0)
        g = gyro[pos:end].unsqueeze(0)
        att = attitude[pos:end].unsqueeze(0)

        if airimu is not None:
            a_in, g_in, lv_imu = airimu.correct(a, g)
            imu_log_var[pos:end] = lv_imu[0].cpu().numpy()
        else:
            a_in, g_in = a, g

        v, lv = airio(a_in, g_in, att)
        v_pred[pos:end] = v[0].cpu().numpy()
        log_var[pos:end] = lv[0].cpu().numpy()
        pos = end

    return v_pred, log_var, imu_log_var


def _run_pipeline(
    seq: _Sequence,
    airio: AirIONet,
    airimu: AirIMUNet | None,
    device: torch.device,
    chunk: int,
    attitude_source: str,
) -> dict[str, np.ndarray]:
    """End-to-end inference on one sequence; returns a trajectory dict."""
    n = seq.n
    dt = seq.dt

    acc_t = torch.from_numpy(seq.acc).to(device)
    gyro_t = torch.from_numpy(seq.gyro).to(device)

    if attitude_source == "gt":
        att_np = seq.attitude
    else:
        # Use raw IMU-integrated attitude as a placeholder. The EKF below
        # then refines it; for the AirIO input we re-derive attitude from
        # the EKF state at chunk boundaries.
        att_np = seq.attitude.copy()
    att_t = torch.from_numpy(att_np).to(device).float()

    v_pred, log_var, imu_log_var = _airio_forward_chunked(
        airio, airimu, acc_t, gyro_t, att_t, chunk
    )

    # ---- EKF dead-reckoning + measurement update per sample ----
    ekf = VelocityEKF()
    ekf.reset(
        R=seq.R[0].copy(),
        v=seq.R[0] @ seq.v_body[0],
        p=seq.p[0].copy(),
    )

    pos_traj = np.zeros((n, 3), dtype=np.float64)
    R_traj = np.zeros((n, 3, 3), dtype=np.float64)
    pos_traj[0] = ekf.p
    R_traj[0] = ekf.R

    # If AirIMU is unavailable we feed raw IMU into the EKF as well.
    acc_np = seq.acc.astype(np.float64)
    gyro_np = seq.gyro.astype(np.float64)
    if airimu is not None:
        # Re-run AirIMU to get the corrected IMU stream once for the EKF.
        with torch.no_grad():
            a_hat, g_hat, _ = airimu.correct(acc_t.unsqueeze(0), gyro_t.unsqueeze(0))
        acc_np = a_hat[0].cpu().numpy().astype(np.float64)
        gyro_np = g_hat[0].cpu().numpy().astype(np.float64)

    rechunk_every = chunk if attitude_source == "ekf" else None
    for i in range(1, n):
        ekf.predict(
            acc=acc_np[i - 1],
            gyro=gyro_np[i - 1],
            dt=dt,
            imu_log_var=imu_log_var[i - 1] if airimu is not None else None,
        )
        ekf.update_velocity(v_body_meas=v_pred[i], log_var=log_var[i])
        pos_traj[i] = ekf.p
        R_traj[i] = ekf.R

        # Optional re-encode of attitude for AirIO — only useful in the
        # ``ekf`` mode where we want AirIO to see the EKF's attitude.
        if (
            rechunk_every is not None
            and (i + 1) % rechunk_every == 0
            and i + 1 < n
        ):
            xi = so3_log_np(R_traj[i])
            att_np[i + 1 :] = xi  # crude: hold last attitude until next chunk
            att_t = torch.from_numpy(att_np).to(device).float()
            v_new, lv_new, _ = _airio_forward_chunked(
                airio, airimu, acc_t, gyro_t, att_t, chunk
            )
            v_pred[i + 1 :] = v_new[i + 1 :]
            log_var[i + 1 :] = lv_new[i + 1 :]

    return {
        "pos_pred": pos_traj,
        "R_pred": R_traj,
        "v_pred": v_pred,
        "pos_gt": seq.p,
        "R_gt": seq.R,
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def _ate_rte_rot(
    out: dict[str, np.ndarray], imu_rate: float, rte_interval: float
) -> dict[str, float]:
    """Compute ATE (Eq. 13), RTE (Eq. 14) and mean rotation error in degrees."""
    pos_pred = out["pos_pred"]
    pos_gt = out["pos_gt"]
    R_pred = out["R_pred"]
    R_gt = out["R_gt"]

    ate = float(np.sqrt(np.mean(np.sum((pos_pred - pos_gt) ** 2, axis=1))))

    step = max(1, int(round(imu_rate * rte_interval)))
    n = pos_pred.shape[0]
    rte_residuals = []
    for i in range(0, n - step):
        gt_disp = pos_gt[i + step] - pos_gt[i]
        gt_disp_local = R_gt[i].T @ gt_disp
        pred_disp = pos_pred[i + step] - pos_pred[i]
        pred_disp_local = R_pred[i].T @ pred_disp
        rte_residuals.append(np.linalg.norm(pred_disp_local - gt_disp_local))
    rte = float(np.sqrt(np.mean(np.square(rte_residuals)))) if rte_residuals else 0.0

    rot_err_rad = []
    for i in range(n):
        R_diff = R_pred[i].T @ R_gt[i]
        cos_a = np.clip((np.trace(R_diff) - 1.0) / 2.0, -1.0, 1.0)
        rot_err_rad.append(float(np.arccos(cos_a)))
    rot_err_rad = np.asarray(rot_err_rad)
    rot_err_deg = float(np.degrees(rot_err_rad).mean())
    return {"ate": ate, "rte": rte, "rot_deg": rot_err_deg, "rot_per_frame_deg":
            np.degrees(rot_err_rad)}


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def _plot_sequence(
    name: str, out: dict[str, np.ndarray], metrics: dict, out_dir: Path
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    pos_pred = out["pos_pred"]
    pos_gt = out["pos_gt"]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(pos_gt[:, 0], pos_gt[:, 1], "b-", label="ground truth")
    ax.plot(pos_pred[:, 0], pos_pred[:, 1], "r-", label="prediction")
    ax.scatter([pos_gt[0, 0]], [pos_gt[0, 1]], c="green", s=60, label="start", zorder=5)
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_title(f"{name} — top-down trajectory")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"{name}_trajectory.png", dpi=150)
    plt.close(fig)

    err = np.linalg.norm(pos_pred - pos_gt, axis=1)
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(err, "r-")
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Position error [m]")
    ax.set_title(f"{name} — per-sample translation error (ATE component)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / f"{name}_trans_error.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(metrics["rot_per_frame_deg"], "r-")
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Rotation error [deg]")
    ax.set_title(f"{name} — per-sample rotation error")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / f"{name}_rot_error.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    print(f"{'sequence':<20} {'ATE [m]':>10} {'RTE [m]':>10} {'rot [deg]':>12} {'samples':>8}")
    print("-" * 64)
    aggregate = {"ate": [], "rte": [], "rot_deg": []}
    with torch.no_grad():
        for name in seq_names:
            seq = _Sequence(Path(args.data_root) / name, imu_rate=args.imu_rate)
            out = _run_pipeline(
                seq, airio, airimu, device,
                chunk=args.airio_chunk,
                attitude_source=args.attitude_source,
            )
            metrics = _ate_rte_rot(out, imu_rate=args.imu_rate,
                                   rte_interval=args.rte_interval)
            print(f"{name:<20} {metrics['ate']:>10.4f} {metrics['rte']:>10.4f} "
                  f"{metrics['rot_deg']:>12.4f} {seq.n:>8d}")
            _plot_sequence(name, out, metrics, out_dir)
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
