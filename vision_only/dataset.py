"""
PyTorch dataset for UAV odometry sequences.

Each sequence directory has the layout produced by the Blender pipeline::

    sequence/
        images/
            000000.png
            000001.png
            ...
        frame_index.csv   # t, imu_index, image_path  (camera-rate)
        poses_body.csv    # t, x, y, z, qw, qx, qy, qz  (IMU-rate)
        imu_clean.csv     # t, gx, gy, gz, ax, ay, az   (IMU-rate, unused here)

The dataset slices each sequence into windows of ``sequence_length`` frame
pairs and returns the relative pose between consecutive frames.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset


_IMAGENET_MEAN = (0.485, 0.456, 0.406)


def _quat_to_rotmat(qw: np.ndarray, qx: np.ndarray,
                    qy: np.ndarray, qz: np.ndarray) -> np.ndarray:
    """Unit quaternions (n,) → rotation matrices (n, 3, 3)."""
    R = np.empty((len(qw), 3, 3), dtype=np.float64)
    R[:, 0, 0] = 1 - 2 * (qy ** 2 + qz ** 2)
    R[:, 0, 1] = 2 * (qx * qy - qz * qw)
    R[:, 0, 2] = 2 * (qx * qz + qy * qw)
    R[:, 1, 0] = 2 * (qx * qy + qz * qw)
    R[:, 1, 1] = 1 - 2 * (qx ** 2 + qz ** 2)
    R[:, 1, 2] = 2 * (qy * qz - qx * qw)
    R[:, 2, 0] = 2 * (qx * qz - qy * qw)
    R[:, 2, 1] = 2 * (qy * qz + qx * qw)
    R[:, 2, 2] = 1 - 2 * (qx ** 2 + qy ** 2)
    return R
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _read_frame_index(root: Path) -> tuple[list[str], np.ndarray]:
    """Return (frame_file_paths, imu_indices) from frame_index.csv."""
    paths = []
    imu_indices = []
    with open(root / "frame_index.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            paths.append(str(root / row["image_path"]))
            imu_indices.append(int(row["imu_index"]))
    return paths, np.array(imu_indices, dtype=np.int64)


def _load_camera_rate_poses(root: Path, imu_indices: np.ndarray) -> np.ndarray:
    """Load poses_body.csv and return 4×4 matrices at camera-rate timestamps."""
    data = np.loadtxt(root / "poses_body.csv", delimiter=",", skiprows=1)
    # columns: t, x, y, z, qw, qx, qy, qz
    xyz = data[imu_indices, 1:4]
    qw = data[imu_indices, 4]
    qx = data[imu_indices, 5]
    qy = data[imu_indices, 6]
    qz = data[imu_indices, 7]
    R = _quat_to_rotmat(qw, qx, qy, qz)  # (n_cam, 3, 3)
    n = len(imu_indices)
    T = np.zeros((n, 4, 4), dtype=np.float64)
    T[:, :3, :3] = R
    T[:, :3, 3] = xyz
    T[:, 3, 3] = 1.0
    return T


class _Sequence:
    """In-memory description of a single trajectory."""

    def __init__(self, root: Path, sequence_length: int) -> None:
        self.root = root
        self.sequence_length = sequence_length

        frame_files, imu_indices = _read_frame_index(root)
        if len(frame_files) < 2:
            raise RuntimeError(f"sequence {root} has fewer than 2 frames")
        self.frame_files = frame_files

        abs_poses = _load_camera_rate_poses(root, imu_indices)
        n_frames = len(self.frame_files)

        self.rel_R = np.zeros((n_frames - 1, 3, 3), dtype=np.float32)
        self.rel_t = np.zeros((n_frames - 1, 3), dtype=np.float32)
        for i in range(n_frames - 1):
            T_rel = np.linalg.inv(abs_poses[i]) @ abs_poses[i + 1]
            self.rel_R[i] = T_rel[:3, :3]
            self.rel_t[i] = T_rel[:3, 3]

        self.num_pairs = n_frames - 1
        self.num_windows = max(0, self.num_pairs - sequence_length + 1)

    def window(self, start: int) -> tuple[list[str], list[str], np.ndarray, np.ndarray]:
        T = self.sequence_length
        files_t = self.frame_files[start : start + T]
        files_t1 = self.frame_files[start + 1 : start + 1 + T]
        rel_R = self.rel_R[start : start + T]
        rel_t = self.rel_t[start : start + T]
        return files_t, files_t1, rel_R, rel_t


class UAVOdometryDataset(Dataset):
    """Dataset of fixed-length windows of frame pairs and relative poses."""

    def __init__(
        self,
        root_dir: str,
        sequence_length: int = 10,
        img_height: int = 224,
        img_width: int = 224,
        augment: bool = False,
        sequences: list[str] | None = None,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.sequence_length = sequence_length
        self.img_height = img_height
        self.img_width = img_width
        self.augment = augment

        if sequences is None:
            seq_dirs = sorted(
                d for d in self.root_dir.iterdir()
                if d.is_dir()
                and (d / "images").exists()
                and (d / "frame_index.csv").exists()
                and (d / "poses_body.csv").exists()
            )
        else:
            seq_dirs = [self.root_dir / name for name in sequences]

        self.sequences: list[_Sequence] = []
        self.index: list[tuple[int, int]] = []
        for seq_dir in seq_dirs:
            seq = _Sequence(seq_dir, sequence_length)
            if seq.num_windows == 0:
                continue
            seq_idx = len(self.sequences)
            self.sequences.append(seq)
            for start in range(seq.num_windows):
                self.index.append((seq_idx, start))

        if not self.index:
            raise RuntimeError(
                f"No usable sequences found under {root_dir} with "
                f"sequence_length={sequence_length}"
            )

    @staticmethod
    def list_sequences(root_dir: str) -> list[str]:
        root = Path(root_dir)
        return sorted(
            d.name for d in root.iterdir()
            if d.is_dir()
            and (d / "images").exists()
            and (d / "frame_index.csv").exists()
            and (d / "poses_body.csv").exists()
        )

    def __len__(self) -> int:
        return len(self.index)

    def _load_image(self, path: str) -> torch.Tensor:
        img = Image.open(path).convert("RGB")
        img = img.resize((self.img_width, self.img_height), Image.BILINEAR)
        return TF.to_tensor(img)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        seq_idx, start = self.index[idx]
        seq = self.sequences[seq_idx]
        files_t, files_t1, rel_R, rel_t = seq.window(start)

        T = self.sequence_length
        frames_t = torch.empty((T, 3, self.img_height, self.img_width), dtype=torch.float32)
        frames_t1 = torch.empty_like(frames_t)

        for i in range(T):
            f_t = self._load_image(files_t[i])
            f_t1 = self._load_image(files_t1[i])
            if self.augment:
                brightness = float(np.random.uniform(0.8, 1.2))
                contrast = float(np.random.uniform(0.8, 1.2))
                f_t = TF.adjust_brightness(f_t, brightness)
                f_t1 = TF.adjust_brightness(f_t1, brightness)
                f_t = TF.adjust_contrast(f_t, contrast)
                f_t1 = TF.adjust_contrast(f_t1, contrast)
            f_t = TF.normalize(f_t, _IMAGENET_MEAN, _IMAGENET_STD)
            f_t1 = TF.normalize(f_t1, _IMAGENET_MEAN, _IMAGENET_STD)
            frames_t[i] = f_t
            frames_t1[i] = f_t1

        R_gt = torch.from_numpy(rel_R.astype(np.float32))
        trans_gt = torch.from_numpy(rel_t.astype(np.float32))
        rot_6d_gt = torch.cat([R_gt[:, :, 0], R_gt[:, :, 1]], dim=-1)

        return {
            "frames_t": frames_t,
            "frames_t1": frames_t1,
            "trans_gt": trans_gt,
            "R_gt": R_gt,
            "rot_6d_gt": rot_6d_gt,
            "sequence": seq.root.name,
            "start": start,
        }
