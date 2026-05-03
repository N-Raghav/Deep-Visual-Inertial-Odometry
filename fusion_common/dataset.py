"""
Paired vision + IMU dataset shared by all three fusion branches.

Each sequence directory has the layout::

    sequence/
        images/
            000000.png
            ...
        frame_index.csv   # t, imu_index, image_path  (camera-rate)
        poses_body.csv    # t, x, y, z, qw, qx, qy, qz  (IMU-rate)
        imu_clean.csv     # t, gx, gy, gz, ax, ay, az   (IMU-rate)

The camera runs at a lower rate than the IMU (e.g. 100 Hz vs 1000 Hz).
``frame_index.csv`` maps each camera frame to its corresponding IMU sample
index, providing the alignment between the two streams.

Primary arrays (``acc``, ``gyro``, ``attitude``, ``v_body``, ``R``, ``p``)
are stored at **IMU rate** — this is what loose_fusion needs for its
sample-by-sample EKF.  Camera-rate views ``R_cam`` and ``p_cam`` are
derived from the IMU-rate arrays via ``frame_imu_indices``.  Gated and
cross-attention branches use these camera-rate views for trajectory
comparison and the ``imu_context`` windows for AirIO input.
"""

from __future__ import annotations

import csv
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from imu_only.utils import (
    attitude_logmap_from_poses,
    body_frame_velocity_from_poses,
    quat_to_rotmat_np,
)


_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _read_frame_index(root: Path) -> tuple[list[str], np.ndarray]:
    """Return (frame_file_paths, imu_indices) from frame_index.csv."""
    paths: list[str] = []
    imu_indices: list[int] = []
    with open(root / "frame_index.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            paths.append(str(root / row["image_path"]))
            imu_indices.append(int(row["imu_index"]))
    return paths, np.array(imu_indices, dtype=np.int64)


def _load_imu_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    gyro = data[:, 1:4].astype(np.float32)  # gx, gy, gz
    acc = data[:, 4:7].astype(np.float32)   # ax, ay, az
    return acc, gyro


def _load_poses_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    p = data[:, 1:4].astype(np.float64)
    R = quat_to_rotmat_np(data[:, 4], data[:, 5], data[:, 6], data[:, 7])
    return R, p


class _PairedSequence:
    """In-memory description of a single trajectory: poses + IMU + frames.

    Attributes at IMU rate (primary, length ``n``):
        acc, gyro, attitude, v_body, R, p, dt

    Camera-rate views (length ``n_cam``):
        frame_files, frame_imu_indices, R_cam, p_cam, dt_cam
    """

    def __init__(self, root: Path, imu_rate: float) -> None:
        self.root = root

        frame_files, frame_imu_indices = _read_frame_index(root)
        acc, gyro = _load_imu_csv(root / "imu_clean.csv")
        R_imu, p_imu = _load_poses_csv(root / "poses_body.csv")

        n_imu = min(len(acc), len(R_imu))
        if len(acc) != len(R_imu):
            warnings.warn(
                f"{root}: imu ({len(acc)}) and poses ({len(R_imu)}) "
                f"differ; truncating to {n_imu}"
            )

        # Clip camera frames to those whose IMU index is in range.
        valid = frame_imu_indices < n_imu
        frame_files = [frame_files[i] for i in range(len(frame_files)) if valid[i]]
        frame_imu_indices = frame_imu_indices[valid]

        self.acc = acc[:n_imu]
        self.gyro = gyro[:n_imu]
        self.R = R_imu[:n_imu].astype(np.float64)
        self.p = p_imu[:n_imu].astype(np.float64)
        self.dt = 1.0 / float(imu_rate)
        self.imu_rate = float(imu_rate)
        self.n = n_imu  # primary count — used by loose_fusion EKF loop

        self.v_body = body_frame_velocity_from_poses(self.R, self.p, self.dt).astype(
            np.float32
        )
        self.attitude = attitude_logmap_from_poses(self.R).astype(np.float32)

        # Camera-rate views.
        self.frame_files = frame_files
        self.frame_imu_indices = frame_imu_indices  # (n_cam,) int64
        self.n_cam = len(frame_files)
        self.R_cam = self.R[frame_imu_indices]   # (n_cam, 3, 3)
        self.p_cam = self.p[frame_imu_indices]   # (n_cam, 3)

        # Camera timestep derived from consecutive frame IMU indices.
        if len(frame_imu_indices) >= 2:
            self.dt_cam = float(frame_imu_indices[1] - frame_imu_indices[0]) * self.dt
        else:
            self.dt_cam = self.dt


class PairedDataset(Dataset):
    """Sliding window of camera-rate frame pairs with IMU-rate context.

    Args:
        root_dir: Top-level dataset directory.
        sequence_length: ``T`` — number of frame pairs per training window.
        imu_context: Number of IMU samples to feed AirIO per frame pair.
            The window ends at the IMU sample aligned with the next frame.
        img_height / img_width: image resize target.
        imu_rate: IMU sample rate in Hz. Defaults to 1000.
        augment: photometric jitter on frame pairs.
        sequences: optional explicit list of sequence directory names.
    """

    def __init__(
        self,
        root_dir: str,
        sequence_length: int = 10,
        imu_context: int = 100,
        img_height: int = 224,
        img_width: int = 224,
        imu_rate: float = 1000.0,
        augment: bool = False,
        sequences: list[str] | None = None,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.sequence_length = int(sequence_length)
        self.imu_context = int(imu_context)
        self.img_height = int(img_height)
        self.img_width = int(img_width)
        self.imu_rate = float(imu_rate)
        self.augment = bool(augment)

        if sequences is None:
            seq_dirs = sorted(
                d for d in self.root_dir.iterdir()
                if d.is_dir()
                and (d / "images").exists()
                and (d / "frame_index.csv").exists()
                and (d / "poses_body.csv").exists()
                and (d / "imu_clean.csv").exists()
            )
        else:
            seq_dirs = [self.root_dir / name for name in sequences]

        self.sequences: list[_PairedSequence] = []
        self.index: list[tuple[int, int]] = []
        for seq_dir in seq_dirs:
            seq = _PairedSequence(seq_dir, imu_rate=self.imu_rate)
            # First camera frame index that has >= imu_context IMU samples of history.
            min_start = int(np.searchsorted(seq.frame_imu_indices, self.imu_context - 1))
            max_start = seq.n_cam - self.sequence_length
            if max_start <= min_start:
                continue
            seq_idx = len(self.sequences)
            self.sequences.append(seq)
            for start in range(min_start, max_start):
                self.index.append((seq_idx, start))

        if not self.index:
            raise RuntimeError(
                f"No usable sequences in {root_dir} for "
                f"sequence_length={sequence_length}, imu_context={imu_context}"
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
            and (d / "imu_clean.csv").exists()
        )

    def __len__(self) -> int:
        return len(self.index)

    def _load_image(self, path: str) -> torch.Tensor:
        img = Image.open(path).convert("RGB").resize(
            (self.img_width, self.img_height), Image.BILINEAR
        )
        return TF.to_tensor(img)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        seq_idx, start = self.index[idx]
        seq = self.sequences[seq_idx]
        T = self.sequence_length
        W = self.imu_context

        frames_t = torch.empty((T, 3, self.img_height, self.img_width), dtype=torch.float32)
        frames_t1 = torch.empty_like(frames_t)
        imu_acc = torch.empty((T, W, 3), dtype=torch.float32)
        imu_gyro = torch.empty((T, W, 3), dtype=torch.float32)
        attitude = torch.empty((T, W, 3), dtype=torch.float32)
        v_body_gt = torch.empty((T, 3), dtype=torch.float32)
        rel_R = torch.empty((T, 3, 3), dtype=torch.float32)
        rel_t = torch.empty((T, 3), dtype=torch.float32)

        for i in range(T):
            t0 = start + i       # camera frame index
            t1 = start + i + 1   # camera frame index

            f0 = self._load_image(seq.frame_files[t0])
            f1 = self._load_image(seq.frame_files[t1])
            if self.augment:
                brightness = float(np.random.uniform(0.8, 1.2))
                contrast = float(np.random.uniform(0.8, 1.2))
                f0 = TF.adjust_brightness(f0, brightness)
                f1 = TF.adjust_brightness(f1, brightness)
                f0 = TF.adjust_contrast(f0, contrast)
                f1 = TF.adjust_contrast(f1, contrast)
            f0 = TF.normalize(f0, _IMAGENET_MEAN, _IMAGENET_STD)
            f1 = TF.normalize(f1, _IMAGENET_MEAN, _IMAGENET_STD)
            frames_t[i] = f0
            frames_t1[i] = f1

            # IMU context: W samples ending at the IMU index of frame t1.
            imu_hi = int(seq.frame_imu_indices[t1])
            imu_lo = imu_hi - W + 1
            imu_acc[i] = torch.from_numpy(seq.acc[imu_lo : imu_hi + 1])
            imu_gyro[i] = torch.from_numpy(seq.gyro[imu_lo : imu_hi + 1])
            attitude[i] = torch.from_numpy(seq.attitude[imu_lo : imu_hi + 1])
            v_body_gt[i] = torch.from_numpy(seq.v_body[imu_hi])

            # Relative pose from camera-rate poses.
            R_rel = seq.R_cam[t0].T @ seq.R_cam[t1]
            t_rel = seq.R_cam[t0].T @ (seq.p_cam[t1] - seq.p_cam[t0])
            rel_R[i] = torch.from_numpy(R_rel.astype(np.float32))
            rel_t[i] = torch.from_numpy(t_rel.astype(np.float32))

        return {
            "frames_t": frames_t,
            "frames_t1": frames_t1,
            "imu_acc": imu_acc,
            "imu_gyro": imu_gyro,
            "attitude": attitude,
            "v_body_gt": v_body_gt,
            "trans_gt": rel_t,
            "R_gt": rel_R,
            "rot_6d_gt": torch.cat([rel_R[..., 0], rel_R[..., 1]], dim=-1),
            "sequence": seq.root.name,
            "start": start,
            "dt": torch.tensor(seq.dt, dtype=torch.float32),
        }
