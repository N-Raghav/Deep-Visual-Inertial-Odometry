"""
Evaluate a trained Branch A checkpoint on held-out trajectories.

For each test sequence the script:

    1. Runs the network with the LSTM hidden state carried across the
       full trajectory.
    2. Reconstructs the predicted trajectory by dead-reckoning the
       per-step relative poses.
    3. Reports ATE (after Umeyama alignment), RTE (mean per-frame
       translation error) and the mean rotation error in degrees.
    4. Saves three plots per sequence into ``--results_dir``:
       top-down trajectory, per-frame translation error, per-frame
       rotation error.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from dataset import UAVOdometryDataset, _Sequence, _IMAGENET_MEAN, _IMAGENET_STD
from model import BranchA
from utils import (
    integrate_trajectory,
    pose_to_matrix,
    rotation_geodesic_deg,
    umeyama_alignment,
)
import torchvision.transforms.functional as TF
from PIL import Image


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate Branch A on test sequences.")
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--results_dir", type=str, default="results/vision_only")
    p.add_argument("--img_height", type=int, default=224)
    p.add_argument("--img_width", type=int, default=224)
    p.add_argument("--chunk_size", type=int, default=10,
                   help="Number of frame pairs fed through the LSTM at once. "
                        "Hidden state is carried across chunks.")
    p.add_argument("--sequences", type=str, nargs="*", default=None,
                   help="Optional explicit list of sequence directory names "
                        "to evaluate. Defaults to all sequences in data_root.")
    return p.parse_args()


def _load_image(path: str, h: int, w: int) -> torch.Tensor:
    """Load and ImageNet-normalize a single image."""
    img = Image.open(path).convert("RGB").resize((w, h), Image.BILINEAR)
    tensor = TF.to_tensor(img)
    tensor = TF.normalize(tensor, _IMAGENET_MEAN, _IMAGENET_STD)
    return tensor


def predict_sequence(
    model: BranchA,
    seq: _Sequence,
    device: torch.device,
    chunk_size: int,
    img_h: int,
    img_w: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict relative ``(R, t)`` for every consecutive frame pair.

    The LSTM hidden state is initialised once and carried across all
    chunks to mimic streaming inference on a real flight.

    Returns:
        Tuple ``(rel_R, rel_t)`` of shapes ``(N, 3, 3)`` and ``(N, 3)``.
    """
    n_pairs = seq.num_pairs
    rel_R = np.zeros((n_pairs, 3, 3), dtype=np.float32)
    rel_t = np.zeros((n_pairs, 3), dtype=np.float32)

    hidden = None
    pos = 0
    while pos < n_pairs:
        end = min(pos + chunk_size, n_pairs)
        T = end - pos
        frames_t = torch.empty((1, T, 3, img_h, img_w), dtype=torch.float32)
        frames_t1 = torch.empty_like(frames_t)
        for i in range(T):
            frames_t[0, i] = _load_image(seq.frame_files[pos + i], img_h, img_w)
            frames_t1[0, i] = _load_image(seq.frame_files[pos + i + 1], img_h, img_w)
        frames_t = frames_t.to(device, non_blocking=True)
        frames_t1 = frames_t1.to(device, non_blocking=True)

        trans_pred, _, R_pred, hidden, _ = model(frames_t, frames_t1, hidden=hidden)
        # Detach hidden so memory does not grow across chunks.
        hidden = (hidden[0].detach(), hidden[1].detach())

        rel_R[pos:end] = R_pred[0].detach().cpu().numpy()
        rel_t[pos:end] = trans_pred[0].detach().cpu().numpy()
        pos = end

    return rel_R, rel_t


def evaluate_sequence(
    model: BranchA,
    seq: _Sequence,
    device: torch.device,
    chunk_size: int,
    img_h: int,
    img_w: int,
    out_dir: Path,
) -> dict[str, float]:
    """Run prediction + metric computation + plotting for one sequence."""
    rel_R_pred, rel_t_pred = predict_sequence(model, seq, device, chunk_size, img_h, img_w)
    rel_R_gt = seq.rel_R
    rel_t_gt = seq.rel_t

    # Dead-reckoned world poses.
    poses_pred = integrate_trajectory(rel_R_pred, rel_t_pred)
    poses_gt = integrate_trajectory(rel_R_gt, rel_t_gt)

    pos_pred = poses_pred[:, :3, 3]
    pos_gt = poses_gt[:, :3, 3]

    # ATE: RMSE after similarity alignment.
    _, _, _, pos_pred_aligned = umeyama_alignment(pos_pred, pos_gt, with_scale=True)
    diffs = pos_pred_aligned - pos_gt
    per_frame_aligned_err = np.linalg.norm(diffs, axis=1)
    ate = float(np.sqrt(np.mean(per_frame_aligned_err ** 2)))

    # RTE: mean per-frame relative translation error (no alignment).
    rte_per_frame = np.linalg.norm(rel_t_pred - rel_t_gt, axis=1)
    rte = float(rte_per_frame.mean())

    # Mean rotation error in degrees.
    rot_err_deg = rotation_geodesic_deg(rel_R_pred, rel_R_gt)
    mean_rot_deg = float(rot_err_deg.mean())

    # ----- plots -----
    out_dir.mkdir(parents=True, exist_ok=True)

    # Top-down (X-Z plane is the natural view for a downward-facing UAV).
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(pos_gt[:, 0], pos_gt[:, 2], "b-", label="ground truth")
    ax.plot(pos_pred_aligned[:, 0], pos_pred_aligned[:, 2], "r-", label="prediction (aligned)")
    ax.scatter([pos_gt[0, 0]], [pos_gt[0, 2]], c="green", s=60, label="start", zorder=5)
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Z [m]")
    ax.set_title(f"{seq.root.name} — top-down trajectory")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"{seq.root.name}_trajectory.png", dpi=150)
    plt.close(fig)

    fig = plt.figure(figsize=(8, 6))
    ax3 = fig.add_subplot(111, projection="3d")
    ax3.plot(pos_gt[:, 0], pos_gt[:, 1], pos_gt[:, 2], "b-", label="ground truth")
    ax3.plot(pos_pred_aligned[:, 0], pos_pred_aligned[:, 1], pos_pred_aligned[:, 2], "r-", label="vision only")
    ax3.scatter([pos_gt[0, 0]], [pos_gt[0, 1]], [pos_gt[0, 2]], c="green", s=60, label="start")
    ax3.set_xlabel("X [m]")
    ax3.set_ylabel("Y [m]")
    ax3.set_zlabel("Z [m]")
    ax3.set_title(f"{seq.root.name} — trajectory (3D)")
    ax3.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"{seq.root.name}_trajectory_3d.png", dpi=150)
    plt.close(fig)

    # Per-frame translation error (relative, unaligned).
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(rte_per_frame, "r-")
    ax.set_xlabel("Frame index")
    ax.set_ylabel("Translation error [m]")
    ax.set_title(f"{seq.root.name} — per-frame translation error (RTE)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / f"{seq.root.name}_trans_error.png", dpi=150)
    plt.close(fig)

    # Per-frame rotation error.
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(rot_err_deg, "r-")
    ax.set_xlabel("Frame index")
    ax.set_ylabel("Rotation error [deg]")
    ax.set_title(f"{seq.root.name} — per-frame rotation error")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / f"{seq.root.name}_rot_error.png", dpi=150)
    plt.close(fig)

    return {"ate": ate, "rte": rte, "rot_deg": mean_rot_deg, "n_pairs": seq.num_pairs}


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = BranchA().to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    state_dict = ckpt.get("model", ckpt)
    model.load_state_dict(state_dict)
    model.eval()

    seq_names = args.sequences or UAVOdometryDataset.list_sequences(args.data_root)
    out_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'sequence':<20} {'ATE [m]':>10} {'RTE [m]':>10} {'rot [deg]':>12} {'pairs':>8}")
    print("-" * 64)

    aggregate = {"ate": [], "rte": [], "rot_deg": []}
    with torch.no_grad():
        for name in seq_names:
            seq = _Sequence(Path(args.data_root) / name, sequence_length=1)
            metrics = evaluate_sequence(
                model, seq, device, args.chunk_size, args.img_height, args.img_width, out_dir
            )
            print(
                f"{name:<20} {metrics['ate']:>10.4f} {metrics['rte']:>10.4f} "
                f"{metrics['rot_deg']:>12.4f} {metrics['n_pairs']:>8d}"
            )
            aggregate["ate"].append(metrics["ate"])
            aggregate["rte"].append(metrics["rte"])
            aggregate["rot_deg"].append(metrics["rot_deg"])

    print("-" * 64)
    print(
        f"{'mean':<20} {np.mean(aggregate['ate']):>10.4f} "
        f"{np.mean(aggregate['rte']):>10.4f} {np.mean(aggregate['rot_deg']):>12.4f}"
    )
    print(f"\nPlots written to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
