"""
Evaluate the gated tight-fusion network.

Loads a trained ``GatedFusionNet`` checkpoint, runs it over each test
sequence, dead-reckons the per-pair relative poses into a world
trajectory, and reports ATE / RTE / mean rotation error using the same
metric module as the other fusion branches.

Plots written to ``--results_dir``:

    {sequence}_trajectory.png   — top-down (X, Y) ground truth vs prediction
    {sequence}_trans_error.png  — per-pair translation error
    {sequence}_rot_error.png    — per-pair rotation error in degrees
    {sequence}_gate.png         — mean gate value over time (interpretability)
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
import torchvision.transforms.functional as TF
from PIL import Image

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from fusion_common.dataset import PairedDataset, _PairedSequence  # noqa: E402
from fusion_common.metrics import trajectory_metrics  # noqa: E402
from model import GatedFusionNet  # noqa: E402

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate gated fusion.")
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--results_dir", type=str, default="results/gated_fusion")
    p.add_argument("--imu_rate", type=float, default=30.0)
    p.add_argument("--imu_context", type=int, default=32)
    p.add_argument("--img_height", type=int, default=224)
    p.add_argument("--img_width", type=int, default=224)
    p.add_argument("--rte_interval", type=float, default=5.0)
    p.add_argument("--chunk", type=int, default=10,
                   help="Number of frame pairs per forward pass.")
    p.add_argument("--sequences", type=str, nargs="*", default=None)
    return p.parse_args()


def _load_image(path: str, h: int, w: int) -> torch.Tensor:
    img = Image.open(path).convert("RGB").resize((w, h), Image.BILINEAR)
    t = TF.to_tensor(img)
    return TF.normalize(t, _IMAGENET_MEAN, _IMAGENET_STD)


def _predict_sequence(
    model: GatedFusionNet,
    seq: _PairedSequence,
    device: torch.device,
    chunk: int,
    imu_context: int,
    img_h: int,
    img_w: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Forward the model across a sequence; returns (trans, R, gate)."""
    n_pairs = seq.n - 1
    rel_t = np.zeros((n_pairs, 3), dtype=np.float64)
    rel_R = np.zeros((n_pairs, 3, 3), dtype=np.float64)
    gate_log = np.zeros((n_pairs,), dtype=np.float64)
    pos = imu_context  # ensure enough IMU history before the first pair
    rel_t_offset = pos
    while pos < n_pairs:
        end = min(pos + chunk, n_pairs)
        T = end - pos
        f0 = torch.empty((1, T, 3, img_h, img_w), dtype=torch.float32)
        f1 = torch.empty_like(f0)
        acc = torch.empty((1, T, imu_context, 3), dtype=torch.float32)
        gyro = torch.empty_like(acc)
        att = torch.empty_like(acc)
        for i in range(T):
            t0 = pos + i
            t1 = pos + i + 1
            f0[0, i] = _load_image(seq.frame_files[t0], img_h, img_w)
            f1[0, i] = _load_image(seq.frame_files[t1], img_h, img_w)
            lo = t1 - imu_context + 1
            hi = t1 + 1
            acc[0, i] = torch.from_numpy(seq.acc[lo:hi])
            gyro[0, i] = torch.from_numpy(seq.gyro[lo:hi])
            att[0, i] = torch.from_numpy(seq.attitude[lo:hi])
        out = model(
            f0.to(device), f1.to(device),
            acc.to(device), gyro.to(device), att.to(device).float(),
        )
        rel_t[pos:end] = out["trans"][0].detach().cpu().numpy()
        rel_R[pos:end] = out["R"][0].detach().cpu().numpy()
        gate_log[pos:end] = out["gate"][0].detach().mean(-1).cpu().numpy()
        pos = end

    # The first ``imu_context`` pairs were skipped because they lacked
    # IMU history. Fill them with identity (caller can crop these).
    rel_R[:rel_t_offset] = np.eye(3)
    return rel_t, rel_R, gate_log


def _integrate(rel_R: np.ndarray, rel_t: np.ndarray, T0: np.ndarray) -> np.ndarray:
    """Dead-reckon relative poses into world poses."""
    T_world = [T0.copy()]
    for i in range(rel_R.shape[0]):
        T_rel = np.eye(4)
        T_rel[:3, :3] = rel_R[i]
        T_rel[:3, 3] = rel_t[i]
        T_world.append(T_world[-1] @ T_rel)
    return np.stack(T_world, axis=0)


def _plot(name: str, world: np.ndarray, gt: dict, metrics: dict,
          gate_log: np.ndarray, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    pos_pred = world[:, :3, 3]
    pos_gt = gt["pos_gt"]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(pos_gt[:, 0], pos_gt[:, 1], "b-", label="ground truth")
    ax.plot(pos_pred[:, 0], pos_pred[:, 1], "r-", label="gated fusion")
    ax.scatter([pos_gt[0, 0]], [pos_gt[0, 1]], c="green", s=60, label="start", zorder=5)
    ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]")
    ax.set_title(f"{name} — trajectory")
    ax.axis("equal"); ax.grid(True, alpha=0.3); ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"{name}_trajectory.png", dpi=150)
    plt.close(fig)

    for kind, series, ylabel in [
        ("trans", metrics["per_frame_trans"], "Position error [m]"),
        ("rot", metrics["per_frame_rot_deg"], "Rotation error [deg]"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(series, "r-")
        ax.set_xlabel("Sample index"); ax.set_ylabel(ylabel)
        ax.set_title(f"{name} — per-frame {kind} error")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / f"{name}_{kind}_error.png", dpi=150)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(gate_log, "g-")
    ax.set_xlabel("Pair index")
    ax.set_ylabel("Mean gate value (1=vision, 0=imu)")
    ax.set_ylim([0, 1])
    ax.set_title(f"{name} — gate trajectory")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / f"{name}_gate.png", dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GatedFusionNet().to(device).eval()
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state.get("model", state))

    seq_names = args.sequences or PairedDataset.list_sequences(args.data_root)
    out_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rte_step = max(1, int(round(args.imu_rate * args.rte_interval)))

    print(f"{'sequence':<20} {'ATE [m]':>10} {'RTE [m]':>10} {'rot [deg]':>12} "
          f"{'pairs':>8}")
    print("-" * 64)
    aggregate = {"ate": [], "rte": [], "rot_deg": []}
    with torch.no_grad():
        for name in seq_names:
            seq = _PairedSequence(Path(args.data_root) / name, imu_rate=args.imu_rate)
            rel_t, rel_R, gate_log = _predict_sequence(
                model, seq, device, args.chunk, args.imu_context,
                args.img_height, args.img_width,
            )
            T0 = np.eye(4); T0[:3, :3] = seq.R[0]; T0[:3, 3] = seq.p[0]
            world = _integrate(rel_R, rel_t, T0)
            n = world.shape[0]
            metrics = trajectory_metrics(
                pos_pred=world[:, :3, 3],
                pos_gt=seq.p[:n],
                R_pred=world[:, :3, :3],
                R_gt=seq.R[:n],
                rte_step=rte_step,
                aligned=False,
            )
            print(f"{name:<20} {metrics['ate']:>10.4f} {metrics['rte']:>10.4f} "
                  f"{metrics['rot_deg']:>12.4f} {n:>8d}")
            _plot(name, world, {"pos_gt": seq.p[:n]}, metrics, gate_log, out_dir)
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
