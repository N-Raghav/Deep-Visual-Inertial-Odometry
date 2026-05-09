import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from scipy.spatial.transform import Rotation

sys.path.append(str(Path(__file__).resolve().parents[1]))

from prgflow_mod.fusion import VIOFusion
from prgflow_mod.model import PRGFlow
from vision_only.utils import rotation_geodesic_deg, umeyama_alignment


def _set_axes_equal(ax, pts: np.ndarray) -> None:
    center = (pts.max(axis=0) + pts.min(axis=0)) / 2
    half = (pts.max(axis=0) - pts.min(axis=0)).max() / 2
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate PRGFlow VIO.")
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--results_dir", type=str, default="results/vio_imp")
    p.add_argument("--beta", type=float, default=0.08)
    p.add_argument("--crop_size", type=int, default=128)
    p.add_argument("--sequences", type=str, nargs="*", default=None)
    return p.parse_args()


def list_sequences(root_dir):
    root = Path(root_dir)
    names = []
    for d in sorted(root.iterdir()):
        if d.is_dir() and (d / "images").exists() and (d / "imu.csv").exists():
            names.append(d.name)
    return names


def load_frame_rows(path):
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "t": float(row["t"]),
                    "imu_index": int(row["imu_index"]),
                    "image_path": row["image_path"],
                }
            )
    return rows


def load_pose_rotations(path):
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    if data.ndim == 1:
        data = data[None, :]
    q_xyzw = data[:, [5, 6, 7, 4]]
    R_wb = Rotation.from_quat(q_xyzw).as_matrix()
    return data[:, 0], data[:, 1:4], R_wb


def load_image(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def run_sequence(seq_dir, model, device, beta, crop_size):
    with open(seq_dir / "meta.json") as f:
        meta = json.load(f)
    K = np.array(meta["K"], dtype=np.float64)
    frame_rows = load_frame_rows(seq_dir / "frame_index.csv")
    imu = np.loadtxt(seq_dir / "imu.csv", delimiter=",", skiprows=1)
    imu_t = imu[:, 0]
    imu_g = imu[:, 1:4]
    imu_a = imu[:, 4:7]
    pose_t, pose_p, pose_R = load_pose_rotations(seq_dir / "poses_cam.csv")

    fusion = VIOFusion(model, K=K, crop_size=crop_size, beta=beta, device=device)
    R0 = pose_R[frame_rows[0]["imu_index"]]
    q0 = Rotation.from_matrix(R0).as_quat()
    q0_wxyz = np.array([q0[3], q0[0], q0[1], q0[2]], dtype=np.float64)
    fusion.reset(q0_wxyz)

    pred_positions = [np.zeros(3, dtype=np.float64)]
    pred_rotations = []
    gt_positions = [pose_p[frame_rows[0]["imu_index"]] - pose_p[frame_rows[0]["imu_index"]]]
    gt_rotations = [pose_R[frame_rows[0]["imu_index"]]]

    prev_imu_idx = frame_rows[0]["imu_index"]
    prev_frame_time = frame_rows[0]["t"]
    first_frame = load_image(seq_dir / frame_rows[0]["image_path"])
    fusion.step(first_frame, [], abs(pose_p[prev_imu_idx, 2]), 0.0)
    pred_rotations.append(fusion.prev_R.copy())

    for row in frame_rows[1:]:
        imu_idx = row["imu_index"]
        samples = []
        start = prev_imu_idx + 1
        end = imu_idx + 1
        for j in range(start, end):
            dt_imu = float(imu_t[j] - imu_t[j - 1]) if j > 0 else 0.0
            samples.append((imu_g[j], imu_a[j], dt_imu))
        frame = load_image(seq_dir / row["image_path"])
        altitude = abs(pose_p[imu_idx, 2])
        dt_frame = float(row["t"] - prev_frame_time)
        out = fusion.step(frame, samples, altitude, dt_frame)
        if out is None:
            pred_positions.append(fusion.position.copy())
            pred_rotations.append(fusion.prev_R.copy())
        else:
            pos, R_now, _ = out
            pred_positions.append(pos.copy())
            pred_rotations.append(R_now.copy())

        gt_positions.append(pose_p[imu_idx] - pose_p[frame_rows[0]["imu_index"]])
        gt_rotations.append(pose_R[imu_idx])
        prev_imu_idx = imu_idx
        prev_frame_time = row["t"]

    pred_positions = np.stack(pred_positions, axis=0)
    gt_positions = np.stack(gt_positions, axis=0)
    pred_rotations = np.stack(pred_rotations, axis=0)
    gt_rotations = np.stack(gt_rotations, axis=0)

    _, _, _, pred_aligned = umeyama_alignment(pred_positions, gt_positions, with_scale=True)
    ate = float(np.sqrt(np.mean(np.sum((pred_aligned - gt_positions) ** 2, axis=1))))
    pred_steps = np.diff(pred_positions, axis=0)
    gt_steps = np.diff(gt_positions, axis=0)
    rte = float(np.mean(np.linalg.norm(pred_steps - gt_steps, axis=1)))
    rot_deg = float(rotation_geodesic_deg(pred_rotations, gt_rotations).mean())
    return {
        "pred": pred_positions,
        "pred_aligned": pred_aligned,
        "gt": gt_positions,
        "ate": ate,
        "rte": rte,
        "rot_deg": rot_deg,
    }


def plot_sequence(name, result, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(result["gt"][:, 0], result["gt"][:, 1], result["gt"][:, 2], label="gt", color="tab:blue")
    ax.plot(
        result["pred_aligned"][:, 0],
        result["pred_aligned"][:, 1],
        result["pred_aligned"][:, 2],
        label="pred aligned",
        color="tab:red",
    )
    ax.scatter(
        [result["gt"][0, 0]], [result["gt"][0, 1]], [result["gt"][0, 2]],
        c="green", s=60, label="start",
    )
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_title(f"{name}\nInput: RGB camera + IMU (acc + gyro)")
    ax.legend()
    _set_axes_equal(ax, np.vstack([result["gt"], result["pred_aligned"]]))
    fig.tight_layout()
    fig.savefig(out_dir / f"{name}_traj3d.png", dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location=device)
    ckpt_args = ckpt.get("args", {})
    model = PRGFlow(
        base_channels=ckpt_args.get("base_channels", 32),
        fire_blocks=ckpt_args.get("fire_blocks", 4),
        dropout=ckpt_args.get("dropout", 0.7),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    seq_names = args.sequences or list_sequences(args.data_root)
    out_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_ate = []
    all_rte = []
    all_rot = []

    print(f"{'sequence':<20} {'ATE [m]':>10} {'RTE [m]':>10} {'rot [deg]':>12}")
    print("-" * 58)
    for name in seq_names:
        seq_dir = Path(args.data_root) / name
        result = run_sequence(seq_dir, model, device, beta=args.beta, crop_size=args.crop_size)
        plot_sequence(name, result, out_dir)
        print(f"{name:<20} {result['ate']:>10.4f} {result['rte']:>10.4f} {result['rot_deg']:>12.4f}")
        all_ate.append(result["ate"])
        all_rte.append(result["rte"])
        all_rot.append(result["rot_deg"])

    print("-" * 58)
    print(f"{'mean':<20} {np.mean(all_ate):>10.4f} {np.mean(all_rte):>10.4f} {np.mean(all_rot):>12.4f}")


if __name__ == "__main__":
    main()
