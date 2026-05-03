"""
PyTorch dataset for IMU-only odometry.

Each sequence directory has the layout::

    sequence/
        imu_clean.csv   # t, gx, gy, gz, ax, ay, az  (IMU-rate, e.g. 1000 Hz)
        poses_body.csv  # t, x, y, z, qw, qx, qy, qz (same rate)

Returned items are sliding windows of ``window_size`` IMU samples.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from utils import (
    GRAVITY_VEC,
    attitude_logmap_from_poses,
    body_frame_velocity_from_poses,
    quat_to_rotmat_np,
)


def _load_imu_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load imu_clean.csv → (acc [N,3], gyro [N,3]) in ax,ay,az / gx,gy,gz order."""
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    # columns: t, gx, gy, gz, ax, ay, az
    gyro = data[:, 1:4].astype(np.float32)
    acc = data[:, 4:7].astype(np.float32)
    return acc, gyro


def _load_poses_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load poses_body.csv → (R [N,3,3], p [N,3]) as float64."""
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    # columns: t, x, y, z, qw, qx, qy, qz
    p = data[:, 1:4].astype(np.float64)
    R = quat_to_rotmat_np(data[:, 4], data[:, 5], data[:, 6], data[:, 7])
    return R, p


class _Sequence:
    """In-memory IMU + pose buffers for one trajectory."""

    def __init__(self, root: Path, imu_rate: float) -> None:
        self.root = root
        acc, gyro = _load_imu_csv(root / "imu_clean.csv")
        R, p = _load_poses_csv(root / "poses_body.csv")

        n = min(len(acc), len(R))
        if len(acc) != len(R):
            warnings.warn(
                f"{root}: imu ({len(acc)}) and poses ({len(R)}) "
                f"have different lengths; truncating to {n}"
            )
        self.acc = acc[:n]
        self.gyro = gyro[:n]
        self.R = R[:n]
        self.p = p[:n]

        self.dt = 1.0 / float(imu_rate)
        self.imu_rate = float(imu_rate)
        self.n = n

        self.v_body = body_frame_velocity_from_poses(self.R, self.p, self.dt).astype(
            np.float32
        )
        self.attitude = attitude_logmap_from_poses(self.R).astype(np.float32)

    def window(self, start: int, length: int) -> dict[str, np.ndarray]:
        end = start + length
        return {
            "acc": self.acc[start:end],
            "gyro": self.gyro[start:end],
            "v_body": self.v_body[start:end],
            "attitude": self.attitude[start:end],
            "R_start": self.R[start],
            "p_start": self.p[start],
            "R_end": self.R[end - 1],
            "p_end": self.p[end - 1],
            "v_world_start": self.R[start] @ self.v_body[start],
            "v_world_end": self.R[end - 1] @ self.v_body[end - 1],
        }


class IMUWindowDataset(Dataset):
    """Sliding-window dataset of IMU samples + ground-truth labels.

    Args:
        root_dir: Directory containing per-sequence subdirectories.
        window_size: Number of IMU samples per window.
        step_size: Sliding stride between windows.
        imu_rate: IMU sample rate in Hz. Defaults to 1000.
        sequences: Optional explicit list of sequence directory names.
    """

    def __init__(
        self,
        root_dir: str,
        window_size: int = 1000,
        step_size: int = 10,
        imu_rate: float = 1000.0,
        sequences: list[str] | None = None,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.window_size = int(window_size)
        self.step_size = int(step_size)
        self.imu_rate = float(imu_rate)
        self.dt = 1.0 / self.imu_rate

        if sequences is None:
            seq_dirs = sorted(
                d for d in self.root_dir.iterdir()
                if d.is_dir()
                and (d / "imu_clean.csv").exists()
                and (d / "poses_body.csv").exists()
            )
        else:
            seq_dirs = [self.root_dir / name for name in sequences]

        self.sequences: list[_Sequence] = []
        self.index: list[tuple[int, int]] = []
        for seq_dir in seq_dirs:
            seq = _Sequence(seq_dir, imu_rate=self.imu_rate)
            n_windows = (seq.n - self.window_size) // self.step_size + 1
            if n_windows <= 0:
                continue
            seq_idx = len(self.sequences)
            self.sequences.append(seq)
            for w in range(n_windows):
                self.index.append((seq_idx, w * self.step_size))

        if not self.index:
            raise RuntimeError(
                f"No usable sequences in {root_dir} for window_size={window_size}"
            )

    @staticmethod
    def list_sequences(root_dir: str) -> list[str]:
        root = Path(root_dir)
        return sorted(
            d.name for d in root.iterdir()
            if d.is_dir()
            and (d / "imu_clean.csv").exists()
            and (d / "poses_body.csv").exists()
        )

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        seq_idx, start = self.index[idx]
        seq = self.sequences[seq_idx]
        w = seq.window(start, self.window_size)
        out = {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in w.items()}
        out["dt"] = torch.tensor(seq.dt, dtype=torch.float32)
        out["sequence"] = seq.root.name
        out["start"] = start
        return out
