"""
Paired vision + IMU dataset shared by all three fusion branches.

Assumes a per-sequence layout identical to the one used by the existing
``vision_only/`` and ``imu_only/`` branches::

    sequence_001/
        frames/
            frame_000000.png
            frame_000001.png
            ...
        poses.txt   # one 4x4 transform per line (16 floats)
        imu.txt     # one IMU sample per line: [ax ay az gx gy gz]

We assume **vision frames and IMU samples are 1:1** (one IMU sample per
frame / per pose). At frame index ``t`` the IMU sample, the pose, and
the image are all from the same instant.

Each ``__getitem__`` returns a sliding window of ``T`` *frame pairs*.
For every pair ``i`` (between absolute time-step ``start + i`` and
``start + i + 1``) the item carries:

    - the two RGB frames (already ImageNet-normalised),
    - an IMU context window of length ``imu_context`` ending at the
      next-frame time-step (so AirIO's GRU has enough history),
    - the per-sample attitude inside that IMU window,
    - the body-frame velocity ground-truth at the next-frame time-step,
    - the relative pose ground-truth ``T_rel = inv(T_t) @ T_t1``.

The fusion branches use the relative pose for supervision and the IMU /
attitude / body-velocity to drive AirIMU + AirIO inside the model.
"""

from __future__ import annotations

import warnings
from glob import glob
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset

# Sibling-branch imports — the fusion entry points add the project root
# to ``sys.path`` before importing this module, so these resolve.
from imu_only.utils import (
    attitude_logmap_from_poses,
    body_frame_velocity_from_poses,
)


_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class _PairedSequence:
    """In-memory description of a single trajectory: poses + IMU + frames."""

    def __init__(self, root: Path, imu_rate: float) -> None:
        self.root = root
        # Poses + IMU.
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

        # Frames.
        frame_files = sorted(glob(str(root / "frames" / "frame_*.png")))
        n = min(poses.shape[0], imu.shape[0], len(frame_files))
        if not (poses.shape[0] == imu.shape[0] == len(frame_files)):
            warnings.warn(
                f"{root}: poses({poses.shape[0]}) imu({imu.shape[0]}) "
                f"frames({len(frame_files)}) differ; truncating to {n}"
            )
        self.frame_files = frame_files[:n]
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


class PairedDataset(Dataset):
    """Sliding window of frame pairs with synchronized IMU context.

    Args:
        root_dir: Top-level dataset directory.
        sequence_length: ``T`` — number of frame pairs per training window.
        imu_context: number of IMU samples to feed AirIO per frame pair.
            Larger values give the IMU GRU more temporal context but cost
            memory. Defaults to 32 (≈1 s at 30 Hz).
        img_height / img_width: image resize target.
        imu_rate: IMU sample rate (Hz). With 1:1 frame:IMU pairing this
            is the same as the camera rate.
        augment: photometric jitter on frame pairs (same jitter applied
            to both frames of a pair to keep the relative pose valid).
        sequences: optional explicit list of sequence directory names.
    """

    def __init__(
        self,
        root_dir: str,
        sequence_length: int = 10,
        imu_context: int = 32,
        img_height: int = 224,
        img_width: int = 224,
        imu_rate: float = 30.0,
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
                and (d / "frames").exists()
                and (d / "poses.txt").exists()
                and (d / "imu.txt").exists()
            )
        else:
            seq_dirs = [self.root_dir / name for name in sequences]

        self.sequences: list[_PairedSequence] = []
        # Each entry is (sequence_index, window_start). The window covers
        # frames [start .. start + T] (T+1 frames produce T pairs).
        self.index: list[tuple[int, int]] = []
        for seq_dir in seq_dirs:
            seq = _PairedSequence(seq_dir, imu_rate=self.imu_rate)
            n_pairs = seq.n - 1
            # Need imu_context history before the first frame of the window.
            min_start = self.imu_context
            max_start = n_pairs - self.sequence_length + 1
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
        """Return sequence directories that have all three required files."""
        root = Path(root_dir)
        return sorted(
            d.name for d in root.iterdir()
            if d.is_dir()
            and (d / "frames").exists()
            and (d / "poses.txt").exists()
            and (d / "imu.txt").exists()
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
        # IMU context per pair — for pair i we use samples [start + i + 1 - W .. start + i + 1).
        imu_acc = torch.empty((T, W, 3), dtype=torch.float32)
        imu_gyro = torch.empty((T, W, 3), dtype=torch.float32)
        attitude = torch.empty((T, W, 3), dtype=torch.float32)
        v_body_gt = torch.empty((T, 3), dtype=torch.float32)
        rel_R = torch.empty((T, 3, 3), dtype=torch.float32)
        rel_t = torch.empty((T, 3), dtype=torch.float32)

        for i in range(T):
            t0 = start + i           # frame at time t
            t1 = start + i + 1       # frame at time t+1

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

            imu_lo = t1 - W + 1   # inclusive
            imu_hi = t1 + 1       # exclusive
            imu_acc[i] = torch.from_numpy(seq.acc[imu_lo:imu_hi])
            imu_gyro[i] = torch.from_numpy(seq.gyro[imu_lo:imu_hi])
            attitude[i] = torch.from_numpy(seq.attitude[imu_lo:imu_hi])
            v_body_gt[i] = torch.from_numpy(seq.v_body[t1])

            T_t = seq.T[t0]
            T_t1 = seq.T[t1]
            T_rel = np.linalg.inv(T_t) @ T_t1
            rel_R[i] = torch.from_numpy(T_rel[:3, :3].astype(np.float32))
            rel_t[i] = torch.from_numpy(T_rel[:3, 3].astype(np.float32))

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
