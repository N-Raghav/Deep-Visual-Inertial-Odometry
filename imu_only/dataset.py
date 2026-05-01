"""
PyTorch dataset for IMU-only odometry on the Blender-rendered UAV data.

Each sequence directory has the same layout used by the vision branch::

    sequence_001/
        frames/             # ignored here
        poses.txt           # one 4x4 homogeneous transform per line
        imu.txt             # one IMU sample per line: [ax ay az gx gy gz]

We assume IMU samples and poses are temporally aligned (one IMU sample per
pose). The IMU rate is configurable (default 200 Hz, matching the AirIO
paper); ``dt`` is derived as ``1 / imu_rate``. If pose and IMU file
lengths differ the loader truncates to the shorter of the two and warns.

Returned items are sliding windows of ``window_size`` IMU samples with
stride ``step_size``. Each window carries everything either AirIMU
(integration loss) or AirIO (per-sample velocity loss) needs.
"""

from __future__ import annotations

import warnings
from glob import glob
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from utils import (
    GRAVITY_VEC,
    attitude_logmap_from_poses,
    body_frame_velocity_from_poses,
)


class _Sequence:
    """In-memory pose / IMU buffers + finite-difference labels for one trajectory."""

    def __init__(self, root: Path, imu_rate: float) -> None:
        self.root = root
        poses = np.loadtxt(root / "poses.txt", dtype=np.float64)
        imu = np.loadtxt(root / "imu.txt", dtype=np.float64)

        if poses.ndim == 1:
            poses = poses[None, :]
        if imu.ndim == 1:
            imu = imu[None, :]
        if poses.shape[1] != 16:
            raise RuntimeError(f"{root}/poses.txt must have 16 floats per line")
        if imu.shape[1] != 6:
            raise RuntimeError(f"{root}/imu.txt must have 6 floats per line")

        n = min(poses.shape[0], imu.shape[0])
        if poses.shape[0] != imu.shape[0]:
            warnings.warn(
                f"{root}: poses ({poses.shape[0]}) and imu ({imu.shape[0]}) "
                f"have different lengths; truncating to {n}"
            )
        self.T = poses[:n].reshape(n, 4, 4)
        self.R = self.T[:, :3, :3].astype(np.float64)
        self.p = self.T[:, :3, 3].astype(np.float64)
        self.acc = imu[:n, :3].astype(np.float32)
        self.gyro = imu[:n, 3:].astype(np.float32)

        self.dt = 1.0 / float(imu_rate)
        self.imu_rate = float(imu_rate)
        self.n = n

        # Labels.
        self.v_body = body_frame_velocity_from_poses(self.R, self.p, self.dt).astype(
            np.float32
        )
        self.attitude = attitude_logmap_from_poses(self.R).astype(np.float32)

    def window(self, start: int, length: int) -> dict[str, np.ndarray]:
        """Return a contiguous window starting at index ``start``."""
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
            # Global-frame velocity at the window endpoints, useful for
            # AirIMU's integration loss.
            "v_world_start": self.R[start] @ self.v_body[start],
            "v_world_end": self.R[end - 1] @ self.v_body[end - 1],
        }


class IMUWindowDataset(Dataset):
    """Sliding-window dataset of IMU samples + ground-truth labels.

    Args:
        root_dir: Directory containing the per-sequence subdirectories.
        window_size: Number of IMU samples per window. The AirIO paper
            uses 1000 (5 s at 200 Hz) for AirIO and a shorter window
            (e.g. 20) for AirIMU's pre-integration loss.
        step_size: Sliding stride between consecutive windows.
        imu_rate: IMU sample rate in Hz. Defaults to 200.
        sequences: Optional explicit list of sequence directory names.
    """

    def __init__(
        self,
        root_dir: str,
        window_size: int = 1000,
        step_size: int = 10,
        imu_rate: float = 200.0,
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
                if d.is_dir() and (d / "imu.txt").exists() and (d / "poses.txt").exists()
            )
        else:
            seq_dirs = [self.root_dir / name for name in sequences]

        self.sequences: list[_Sequence] = []
        # Each entry is (sequence_index, window_start_index).
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
        """Names of sequence directories with usable IMU + poses."""
        root = Path(root_dir)
        return sorted(
            d.name for d in root.iterdir()
            if d.is_dir() and (d / "imu.txt").exists() and (d / "poses.txt").exists()
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
