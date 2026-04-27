"""
PyTorch dataset for the Blender-rendered UAV odometry sequences.

Each top-level sequence directory looks like::

    sequence_001/
        frames/
            frame_000000.png
            frame_000001.png
            ...
        poses.txt   # one 4x4 homogeneous transform per line (16 floats)
        imu.txt     # one IMU reading per line: [ax ay az gx gy gz]

The dataset slices each sequence into windows of ``sequence_length`` frame
pairs and returns the relative pose between consecutive frames inside
the window.
"""

from __future__ import annotations

import os
from glob import glob
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset


_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class _Sequence:
    """In-memory description of a single trajectory.

    The class precomputes the relative pose between consecutive frames and
    the list of valid window starting indices, so ``__getitem__`` only has
    to load frames and return slices.
    """

    def __init__(self, root: Path, sequence_length: int) -> None:
        self.root = root
        self.sequence_length = sequence_length

        frame_files = sorted(glob(str(root / "frames" / "frame_*.png")))
        if len(frame_files) < 2:
            raise RuntimeError(f"sequence {root} has fewer than 2 frames")
        self.frame_files = frame_files

        # Load absolute poses as 4x4 matrices.
        poses = np.loadtxt(root / "poses.txt", dtype=np.float64)
        if poses.ndim == 1:
            poses = poses[None, :]
        if poses.shape[1] != 16:
            raise RuntimeError(
                f"poses.txt in {root} must have 16 floats per line, got {poses.shape[1]}"
            )
        self.abs_poses = poses.reshape(-1, 4, 4)

        n_frames = min(len(self.frame_files), self.abs_poses.shape[0])
        self.frame_files = self.frame_files[:n_frames]
        self.abs_poses = self.abs_poses[:n_frames]

        # Relative poses between consecutive frames.
        self.rel_R = np.zeros((n_frames - 1, 3, 3), dtype=np.float32)
        self.rel_t = np.zeros((n_frames - 1, 3), dtype=np.float32)
        for i in range(n_frames - 1):
            T_t = self.abs_poses[i]
            T_t1 = self.abs_poses[i + 1]
            T_rel = np.linalg.inv(T_t) @ T_t1
            self.rel_R[i] = T_rel[:3, :3]
            self.rel_t[i] = T_rel[:3, 3]

        self.num_pairs = n_frames - 1
        # Number of length-T windows that fit into this sequence.
        self.num_windows = max(0, self.num_pairs - sequence_length + 1)

    def window(self, start: int) -> tuple[list[str], list[str], np.ndarray, np.ndarray]:
        """Return the file paths and ground-truth pose for one window."""
        T = self.sequence_length
        files_t = self.frame_files[start : start + T]
        files_t1 = self.frame_files[start + 1 : start + 1 + T]
        rel_R = self.rel_R[start : start + T]
        rel_t = self.rel_t[start : start + T]
        return files_t, files_t1, rel_R, rel_t


class UAVOdometryDataset(Dataset):
    """Dataset of fixed-length windows of frame pairs and relative poses.

    Args:
        root_dir: Directory containing the per-sequence subdirectories.
        sequence_length: ``T`` in the network forward pass — the number of
            frame pairs in one training window.
        img_height: Image height after resize.
        img_width: Image width after resize.
        augment: If ``True`` apply photometric jitter (brightness +
            contrast) consistently to both frames in a pair. No geometric
            augmentation is ever applied because that would change the
            ground-truth relative pose.
        sequences: Optional explicit list of sequence directory names to
            include (used by ``train.py`` for the train/val split).
    """

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
                d for d in self.root_dir.iterdir() if d.is_dir() and (d / "frames").exists()
            )
        else:
            seq_dirs = [self.root_dir / name for name in sequences]

        self.sequences: list[_Sequence] = []
        # Each entry is (sequence_index, window_start_index).
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

    # ------------------------------------------------------------------
    @staticmethod
    def list_sequences(root_dir: str) -> list[str]:
        """Return the sorted list of sequence directory names under ``root_dir``."""
        root = Path(root_dir)
        return sorted(
            d.name for d in root.iterdir() if d.is_dir() and (d / "frames").exists()
        )

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.index)

    # ------------------------------------------------------------------
    def _load_image(self, path: str) -> torch.Tensor:
        """Load an image, resize, and convert to a float tensor in ``[0, 1]``."""
        img = Image.open(path).convert("RGB")
        img = img.resize((self.img_width, self.img_height), Image.BILINEAR)
        tensor = TF.to_tensor(img)  # [3, H, W] in [0, 1]
        return tensor

    # ------------------------------------------------------------------
    def __getitem__(self, idx: int) -> dict[str, Any]:
        seq_idx, start = self.index[idx]
        seq = self.sequences[seq_idx]
        files_t, files_t1, rel_R, rel_t = seq.window(start)

        T = self.sequence_length
        frames_t = torch.empty((T, 3, self.img_height, self.img_width), dtype=torch.float32)
        frames_t1 = torch.empty_like(frames_t)

        # Sample one augmentation per pair, applied identically to both
        # frames of the pair so that the relative pose stays valid.
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

        R_gt = torch.from_numpy(rel_R.astype(np.float32))           # [T, 3, 3]
        trans_gt = torch.from_numpy(rel_t.astype(np.float32))       # [T, 3]
        # 6D ground truth = first two columns of R_gt, flattened.
        rot_6d_gt = torch.cat([R_gt[:, :, 0], R_gt[:, :, 1]], dim=-1)  # [T, 6]

        return {
            "frames_t": frames_t,
            "frames_t1": frames_t1,
            "trans_gt": trans_gt,
            "R_gt": R_gt,
            "rot_6d_gt": rot_6d_gt,
            "sequence": seq.root.name,
            "start": start,
        }
