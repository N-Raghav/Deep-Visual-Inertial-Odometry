"""
Smoke test for the loose-fusion pipeline.

Generates a tiny synthetic dataset matching the layout the
``PairedDataset`` expects (frames + IMU + poses), instantiates the three
backbones with random weights, and runs ``evaluate``-style code on the
synthetic sequence end-to-end. Verifies that the FusionEKF stays finite
and PSD with both vision and IMU updates active.

Usage::

    python test_pipeline.py
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# Sibling-branch imports.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from fusion_common.dataset import _PairedSequence  # noqa: E402
from fusion_ekf import FusionEKF  # noqa: E402
from imu_only.model import AirIMUNet, AirIONet  # noqa: E402
from imu_only.utils import GRAVITY, so3_exp_np  # noqa: E402
from vision_only.model import BranchA  # noqa: E402


_OK = "\033[32mOK\033[0m"
_FAIL = "\033[31mFAIL\033[0m"


def _check(name: str, cond: bool, detail: str = "") -> None:
    tag = _OK if cond else _FAIL
    print(f"  [{tag}] {name}{(' — ' + detail) if detail else ''}")
    if not cond:
        sys.exit(1)


def _make_synthetic_dataset(root: Path, n_seq: int, n_samples: int, imu_rate: float) -> None:
    """Helical trajectory with synchronized frames + IMU + poses (1:1 alignment)."""
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
        R = np.eye(3)
        p = np.array([radius, 0.0, 0.0])
        v_world = np.array([0.0, radius * omega_z, z_rate])
        omega_body = np.array([0.0, 0.0, omega_z])
        base = rng.integers(0, 255, size=(96, 96, 3), dtype=np.uint8)

        for i in range(n_samples):
            poses[i, :3, :3] = R
            poses[i, :3, 3] = p
            poses[i, 3, 3] = 1.0
            theta = omega_z * i * dt
            a_world = np.array(
                [-omega_z ** 2 * radius * np.cos(theta),
                 -omega_z ** 2 * radius * np.sin(theta),
                 0.0]
            )
            spec_world = a_world - np.array([0.0, 0.0, -GRAVITY])
            imu[i, :3] = R.T @ spec_world
            imu[i, 3:] = omega_body

            # Slightly varying frame so vision actually sees motion.
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
    print(" Loose-fusion smoke test")
    print("=" * 60)
    print(f"torch       : {torch.__version__}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tmp = Path(tempfile.mkdtemp(prefix="loose_fusion_smoke_"))
    print(f"tmp dataset : {tmp}")
    try:
        imu_rate = 30.0
        _make_synthetic_dataset(tmp, n_seq=2, n_samples=120, imu_rate=imu_rate)

        # Build backbones with random weights — we just want to confirm
        # the wiring works; numerical accuracy is not the goal here.
        branch_a = BranchA().to(device).eval()
        airimu = AirIMUNet().to(device).eval()
        airio = AirIONet().to(device).eval()
        _check("backbones constructed", True)

        seq = _PairedSequence(tmp / "sequence_000", imu_rate=imu_rate)

        # ----- vision predictions -----
        from PIL import Image as _PIL
        import torchvision.transforms.functional as TF

        def load(p: str) -> torch.Tensor:
            img = _PIL.open(p).convert("RGB").resize((64, 64), _PIL.BILINEAR)
            t = TF.to_tensor(img)
            return TF.normalize(t, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

        n_pairs = seq.n - 1
        f0 = torch.stack(
            [load(seq.frame_files[i]) for i in range(n_pairs)], dim=0
        ).unsqueeze(0).to(device)
        f1 = torch.stack(
            [load(seq.frame_files[i + 1]) for i in range(n_pairs)], dim=0
        ).unsqueeze(0).to(device)
        with torch.no_grad():
            trans, _, R_vis, *_ = branch_a(f0, f1, hidden=None)
        _check("vision output shapes", trans.shape == (1, n_pairs, 3))

        # ----- IMU predictions -----
        with torch.no_grad():
            a_t = torch.from_numpy(seq.acc).to(device).unsqueeze(0)
            g_t = torch.from_numpy(seq.gyro).to(device).unsqueeze(0)
            att_t = torch.from_numpy(seq.attitude).to(device).float().unsqueeze(0)
            a_hat, g_hat, lv_imu = airimu.correct(a_t, g_t)
            v_pred, lv_v = airio(a_hat, g_hat, att_t)
        _check("AirIMU output shape", lv_imu.shape == (1, seq.n, 6))
        _check("AirIO output shape",  v_pred.shape == (1, seq.n, 3))

        # ----- FusionEKF integration -----
        ekf = FusionEKF()
        ekf.reset(R=seq.R[0], v=seq.R[0] @ seq.v_body[0], p=seq.p[0])
        rel_t_vis = trans[0].cpu().numpy()
        rel_R_vis = R_vis[0].cpu().numpy()
        v_pred_np = v_pred[0].cpu().numpy()
        lv_v_np = lv_v[0].cpu().numpy()
        lv_imu_np = lv_imu[0].cpu().numpy()
        a_hat_np = a_hat[0].cpu().numpy().astype(np.float64)
        g_hat_np = g_hat[0].cpu().numpy().astype(np.float64)

        for i in range(1, seq.n):
            ekf.predict(
                acc=a_hat_np[i - 1], gyro=g_hat_np[i - 1],
                dt=seq.dt, imu_log_var=lv_imu_np[i - 1],
            )
            ekf.update_velocity(v_body_meas=v_pred_np[i].astype(np.float64),
                                log_var=lv_v_np[i].astype(np.float64))
            ekf.update_vision_velocity(
                delta_t_vis=rel_t_vis[i - 1].astype(np.float64),
                dt_frame=seq.dt, sigma=0.05,
            )
            ekf.update_vision_rotation(
                delta_R_vis=rel_R_vis[i - 1].astype(np.float64),
                sigma_deg=0.5,
            )

        _check("EKF state finite", np.all(np.isfinite(ekf.p))
                                   and np.all(np.isfinite(ekf.R)))
        eigs = np.linalg.eigvalsh(0.5 * (ekf.P + ekf.P.T))
        _check("EKF covariance PSD", float(eigs.min()) > -1e-6,
               f"min eig={eigs.min():.2e}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("\n" + "=" * 60)
    print(" All checks passed.")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
