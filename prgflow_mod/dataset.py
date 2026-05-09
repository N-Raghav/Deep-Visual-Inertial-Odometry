import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def list_trajectories(root_dir):
    root = Path(root_dir)
    names = []
    for d in sorted(root.iterdir()):
        if d.is_dir() and (d / "pairs").exists():
            names.append(d.name)
    return names


def split_trajectories(root_dir, seed=0):
    names = list_trajectories(root_dir)
    rng = random.Random(seed)
    rng.shuffle(names)
    n = len(names)
    n_train = max(1, int(round(0.8 * n)))
    n_val = int(round(0.1 * n))
    if n_train + n_val > n:
        n_val = max(0, n - n_train)
    n_test = n - n_train - n_val
    if n_test == 0 and n >= 3:
        n_test = 1
        n_train -= 1
    if n_val == 0 and n >= 4:
        n_val = 1
        n_train -= 1
    train = sorted(names[:n_train])
    val = sorted(names[n_train : n_train + n_val])
    test = sorted(names[n_train + n_val :])
    return {"train": train, "val": val, "test": test}


class VIOPairsDataset(Dataset):
    def __init__(self, root_dir, split="train", trajectories=None, seed=0):
        self.root_dir = Path(root_dir)
        if trajectories is None:
            splits = split_trajectories(root_dir, seed=seed)
            trajectories = splits.get(split, [])
        self.trajectories = trajectories
        self.files = []
        for traj in trajectories:
            self.files.extend(sorted((self.root_dir / traj / "pairs").glob("*.npz")))
        if not self.files:
            raise RuntimeError(f"no pair files found for split={split}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        path = self.files[index]
        try:
            data = np.load(path)
        except Exception as e:
            raise RuntimeError(f"error loading {path}") from e
        patch_t = torch.from_numpy(data["patch_t"]).float()
        patch_t1 = torch.from_numpy(data["patch_t1"]).float()
        if patch_t.ndim == 2:
            patch_t = patch_t.unsqueeze(0)
        if patch_t1.ndim == 2:
            patch_t1 = patch_t1.unsqueeze(0)
        label = torch.from_numpy(data["label"]).float()
        dt = torch.tensor(float(data["dt"]), dtype=torch.float32)
        h_t = torch.tensor(float(data["h_t"]), dtype=torch.float32)
        return {
            "patch_t": patch_t,
            "patch_t1": patch_t1,
            "label": label,
            "dt": dt,
            "h_t": h_t,
            "sequence": path.parent.parent.name,
            "path": str(path),
        }
