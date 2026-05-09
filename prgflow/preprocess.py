import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from scipy.spatial.transform import Rotation

from prgflow.warp import warp_image


def parse_args():
    p = argparse.ArgumentParser(description="Preprocess rendered trajectories for PRGFlow VIO.")
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--out_root", type=str, required=True)
    p.add_argument("--crop_size", type=int, default=128)
    p.add_argument("--pad", type=int, default=0)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def list_trajectories(root):
    root = Path(root)
    out = []
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        if (d / "images").exists() and (d / "poses_cam.csv").exists() and (d / "frame_index.csv").exists():
            out.append(d)
    return out


def load_poses(path):
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    if data.ndim == 1:
        data = data[None, :]
    t = data[:, 0]
    p = data[:, 1:4]
    q_xyzw = data[:, [5, 6, 7, 4]]
    R_wb = Rotation.from_quat(q_xyzw).as_matrix()
    return t, p, R_wb


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


def load_image_tensor(path):
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def gray_center_crop(img, crop_size):
    if img.ndim == 3:
        img = img.unsqueeze(0)
    gray = TF.rgb_to_grayscale(img)
    H = gray.shape[-2]
    W = gray.shape[-1]
    top = max(0, (H - crop_size) // 2)
    left = max(0, (W - crop_size) // 2)
    return gray[:, :, top : top + crop_size, left : left + crop_size]


def process_one(traj_dir, out_dir, crop_size, pad, overwrite):
    out_pairs = out_dir / "pairs"
    if out_pairs.exists() and any(out_pairs.glob("*.npz")) and not overwrite:
        print(f"skip {traj_dir.name} (pairs already exist)")
        return

    out_pairs.mkdir(parents=True, exist_ok=True)

    with open(traj_dir / "meta.json") as f:
        meta = json.load(f)
    K = np.array(meta["K"], dtype=np.float64)
    if pad > 0:
        K = K.copy()
        K[0, 2] += pad
        K[1, 2] += pad
    focal = 0.5 * (K[0, 0] + K[1, 1])

    pose_t, pose_p, pose_R = load_poses(traj_dir / "poses_cam.csv")
    frame_rows = load_frame_rows(traj_dir / "frame_index.csv")
    if len(frame_rows) < 2:
        print(f"skip {traj_dir.name} (too few frames)")
        return

    prev_row = frame_rows[0]
    from pathlib import Path
    
    prev_path = traj_dir / prev_row["image_path"]
    
    n_written = 0
    K_inv = np.linalg.inv(K)
    for i in range(len(frame_rows) - 1):
        row_t = frame_rows[i]
        row_t1 = frame_rows[i + 1]

        curr_path = traj_dir / row_t["image_path"]
        next_path = traj_dir / row_t1["image_path"]
        
        if not curr_path.exists() or not next_path.exists():
            continue
            
        img_t = load_image_tensor(curr_path)
        img_t1 = load_image_tensor(next_path)
        
        if pad > 0:
            img_t = F.pad(img_t, (pad, pad, pad, pad))
            img_t1 = F.pad(img_t1, (pad, pad, pad, pad))

        idx_t = row_t["imu_index"]
        idx_t1 = row_t1["imu_index"]
        R_t = pose_R[idx_t]
        R_t1 = pose_R[idx_t1]
        p_t = pose_p[idx_t]
        p_t1 = pose_p[idx_t1]

        H_R = K @ (R_t.T @ R_t1) @ K_inv
        H_R_torch = torch.from_numpy(H_R.astype(np.float32)).unsqueeze(0)
        img_t1_rc = warp_image(img_t1, H_R_torch, out_size=img_t.shape[-2:])

        patch_t = gray_center_crop(img_t, crop_size)[0].numpy().astype(np.float32)
        patch_t1 = gray_center_crop(img_t1_rc, crop_size)[0].numpy().astype(np.float32)

        delta_body = R_t.T @ (p_t1 - p_t)
        h_t = float(abs(p_t[2]))
        if h_t < 1e-6:
            continue

        tx = float(focal * delta_body[0] / h_t)
        ty = float(focal * delta_body[1] / h_t)
        s = float(-delta_body[2] / h_t)
        label = np.array([s, tx, ty], dtype=np.float32)
        dt = float(row_t1["t"] - row_t["t"])

        np.savez_compressed(
            out_pairs / f"{i:06d}.npz",
            patch_t=patch_t,
            patch_t1=patch_t1,
            label=label,
            dt=np.float32(dt),
            h_t=np.float32(h_t),
            frame_t=np.int32(idx_t),
            frame_t1=np.int32(idx_t1),
        )
        n_written += 1

    with open(out_dir / "meta.json", "w") as f:
        json.dump(
            {
                "source": str(traj_dir),
                "crop_size": crop_size,
                "pad": pad,
                "num_pairs": n_written,
                "focal": float(focal),
            },
            f,
            indent=2,
        )
    print(f"{traj_dir.name}: wrote {n_written} pairs")


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    for traj_dir in list_trajectories(data_root):
        process_one(
            traj_dir,
            out_root / traj_dir.name,
            crop_size=args.crop_size,
            pad=args.pad,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
