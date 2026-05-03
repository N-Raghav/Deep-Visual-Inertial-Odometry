"""
Smoke test for the cross-attention tight-fusion network.

Mirrors ``gated_fusion/test_pipeline.py`` but exercises
``CrossAttentionFusionNet`` instead. Same synthetic helical dataset,
same shape and finiteness checks, plus a sanity check that the model's
diagnostic ``vis_imu_cos`` lies in [-1, 1].
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from fusion_common.dataset import PairedDataset  # noqa: E402
from fusion_common.loss import pose_loss  # noqa: E402
from imu_only.utils import GRAVITY, so3_exp_np  # noqa: E402
from model import CrossAttentionFusionNet  # noqa: E402


_OK = "\033[32mOK\033[0m"
_FAIL = "\033[31mFAIL\033[0m"


def _check(name: str, cond: bool, detail: str = "") -> None:
    tag = _OK if cond else _FAIL
    print(f"  [{tag}] {name}{(' — ' + detail) if detail else ''}")
    if not cond:
        sys.exit(1)


def _make_synthetic_dataset(root: Path, n_seq: int, n_samples: int, imu_rate: float) -> None:
    dt = 1.0 / imu_rate
    rng = np.random.default_rng(0)
    for s in range(n_seq):
        seq = root / f"sequence_{s:03d}"
        (seq / "frames").mkdir(parents=True, exist_ok=True)
        omega_z = 0.5 + 0.1 * s
        radius = 1.0 + 0.5 * s
        z_rate = 0.05
        poses = np.zeros((n_samples, 4, 4))
        imu = np.zeros((n_samples, 6))
        R = np.eye(3); p = np.array([radius, 0.0, 0.0])
        v_world = np.array([0.0, radius * omega_z, z_rate])
        omega_body = np.array([0.0, 0.0, omega_z])
        base = rng.integers(0, 255, size=(96, 96, 3), dtype=np.uint8)
        for i in range(n_samples):
            poses[i, :3, :3] = R; poses[i, :3, 3] = p; poses[i, 3, 3] = 1.0
            theta = omega_z * i * dt
            a_world = np.array(
                [-omega_z ** 2 * radius * np.cos(theta),
                 -omega_z ** 2 * radius * np.sin(theta),
                 0.0]
            )
            imu[i, :3] = R.T @ (a_world - np.array([0.0, 0.0, -GRAVITY]))
            imu[i, 3:] = omega_body
            shift = (i * 2) % 16
            patch = base[shift : shift + 64, shift : shift + 64, :].copy()
            Image.fromarray(patch).save(seq / "frames" / f"frame_{i:06d}.png")
            R = R @ so3_exp_np(omega_body * dt)
            p = p + v_world * dt
            v_world = v_world + a_world * dt
        np.savetxt(seq / "poses.txt", poses.reshape(n_samples, 16))
        np.savetxt(seq / "imu.txt", imu)


def main() -> int:
    print("=" * 60)
    print(" Cross-attention tight-fusion smoke test")
    print("=" * 60)
    print(f"torch       : {torch.__version__}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tmp = Path(tempfile.mkdtemp(prefix="cross_attn_smoke_"))
    print(f"tmp dataset : {tmp}")
    try:
        _make_synthetic_dataset(tmp, n_seq=2, n_samples=60, imu_rate=30.0)
        ds = PairedDataset(
            root_dir=str(tmp),
            sequence_length=4, imu_context=16,
            img_height=64, img_width=64,
            imu_rate=30.0, augment=False,
        )
        _check("dataset non-empty", len(ds) > 0, f"len={len(ds)}")
        loader = torch.utils.data.DataLoader(ds, batch_size=2, shuffle=False)
        batch = next(iter(loader))

        net = CrossAttentionFusionNet(
            feat_dim=64, num_heads=2, num_layers=1, ffn_hidden=128, dropout=0.0,
        ).to(device)
        out = net(
            batch["frames_t"].to(device),
            batch["frames_t1"].to(device),
            batch["imu_acc"].to(device),
            batch["imu_gyro"].to(device),
            batch["attitude"].to(device).float(),
        )
        _check("trans shape", out["trans"].shape == (2, 4, 3))
        _check("R shape", out["R"].shape == (2, 4, 3, 3))
        cos = out["vis_imu_cos"]
        _check("vis_imu_cos in [-1, 1]",
               (cos.min().item() >= -1.001) and (cos.max().item() <= 1.001),
               f"min={cos.min():.3f} max={cos.max():.3f}")

        opt = torch.optim.Adam(net.parameters(), lr=1e-4)
        opt.zero_grad()
        total, _, _ = pose_loss(
            out["trans"], batch["trans_gt"].to(device),
            out["R"], batch["R_gt"].to(device),
            lambda_rot=100.0,
        )
        _check("loss finite", torch.isfinite(total).item(),
               f"total={total.item():.4f}")
        total.backward()
        gn = torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        _check("grad finite", torch.isfinite(torch.tensor(float(gn))).item())
        opt.step()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("\n" + "=" * 60)
    print(" All checks passed.")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
