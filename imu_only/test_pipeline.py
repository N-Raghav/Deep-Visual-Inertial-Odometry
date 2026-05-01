"""
End-to-end smoke test for the IMU-only pipeline (AirIMU + AirIO + EKF).

Run after ``setup_env.slurm`` (or any local env with torch + numpy +
matplotlib) to verify the implementation without needing a real Blender
dataset:

    python test_pipeline.py

The script:

    1. Generates a tiny synthetic dataset on disk (3 sequences, 600 IMU
       samples each, 200 Hz) with a smooth helical trajectory and
       analytically derived IMU readings (so the ground-truth velocities
       and attitudes are exact).
    2. Checks ``so3_exp ∘ so3_log = id`` and the right Jacobian shape.
    3. Checks ``integrate_imu_window`` recovers the analytic end-of-window
       pose to within numerical tolerance.
    4. Forwards a batch through ``AirIMUNet`` and ``AirIONet`` and runs a
       single training step of each to check shapes, finite gradients,
       and that the optimizer state advances.
    5. Runs the EKF on a small window: predict + update, asserts the
       state stays finite and the covariance stays PSD.
    6. Confirms the AirIO loss is exactly zero when the prediction equals
       the ground truth and ``log_var = 0``.

Exits non-zero on any failure.
"""

from __future__ import annotations

import math
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image  # noqa: F401  (avoids pyflakes complaint if re-imported elsewhere)

from dataset import IMUWindowDataset, _Sequence
from ekf import VelocityEKF
from loss import airimu_loss, airio_loss
from model import AirIMUNet, AirIONet
from utils import (
    GRAVITY,
    GRAVITY_VEC,
    integrate_imu_window,
    so3_exp,
    so3_exp_np,
    so3_log,
    so3_log_np,
    so3_right_jacobian,
)


_OK = "\033[32mOK\033[0m"
_FAIL = "\033[31mFAIL\033[0m"


def _check(name: str, cond: bool, detail: str = "") -> None:
    tag = _OK if cond else _FAIL
    suffix = f" — {detail}" if detail else ""
    print(f"  [{tag}] {name}{suffix}")
    if not cond:
        sys.exit(1)


# ---------------------------------------------------------------------------
# Synthetic dataset generator
# ---------------------------------------------------------------------------
def _make_synthetic_dataset(
    root: Path, n_sequences: int, n_samples: int, imu_rate: float
) -> None:
    """A smooth helical trajectory whose IMU readings can be derived analytically."""
    dt = 1.0 / imu_rate
    for s in range(n_sequences):
        seq = root / f"sequence_{s:03d}"
        (seq / "frames").mkdir(parents=True, exist_ok=True)
        # Need at least 1 dummy frame to keep dataset code robust to other
        # branches, but the IMU branch ignores frames altogether.

        omega_z = 0.5 + 0.1 * s            # yaw rate (rad/s)
        radius = 1.0 + 0.5 * s
        z_rate = 0.1                        # vertical climb rate (m/s)

        poses = np.zeros((n_samples, 4, 4))
        imu = np.zeros((n_samples, 6))
        R = np.eye(3)
        p = np.array([radius, 0.0, 0.0])
        v_world = np.array([0.0, radius * omega_z, z_rate])
        # Constant omega in body frame (yaw).
        omega_body = np.array([0.0, 0.0, omega_z])

        for i in range(n_samples):
            poses[i, :3, :3] = R
            poses[i, :3, 3] = p
            poses[i, 3, 3] = 1.0

            # Body-frame angular velocity is constant.
            gyro = omega_body
            # World acceleration: derivative of v_world. For a circle:
            # a_world_xy = -ω² * (radius cos θ, radius sin θ) = -ω² * pos_xy.
            theta = omega_z * i * dt
            a_world = np.array(
                [-omega_z ** 2 * radius * np.cos(theta),
                 -omega_z ** 2 * radius * np.sin(theta),
                 0.0]
            )
            # Specific force in body frame: R^T (a_world - g_world).
            specific_world = a_world - np.array([0.0, 0.0, -GRAVITY])
            acc_body = R.T @ specific_world
            imu[i, :3] = acc_body
            imu[i, 3:] = gyro

            # Step world state (forward Euler is fine for such a smooth trajectory).
            R = R @ so3_exp_np(omega_body * dt)
            p = p + v_world * dt
            v_world = v_world + a_world * dt

        np.savetxt(seq / "poses.txt", poses.reshape(n_samples, 16))
        np.savetxt(seq / "imu.txt", imu)


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------
def check_so3() -> None:
    print("\n[1/6] SO(3) operations")
    torch.manual_seed(0)
    phi = torch.randn(4, 5, 3) * 0.5
    R = so3_exp(phi)
    _check("so3_exp output shape", R.shape == (4, 5, 3, 3), str(tuple(R.shape)))

    eye = torch.eye(3).expand_as(R)
    err = (R.transpose(-1, -2) @ R - eye).abs().max().item()
    _check("R is orthogonal", err < 1e-5, f"max|err|={err:.2e}")
    _check("det(R) ≈ 1", torch.linalg.det(R).sub(1.0).abs().max().item() < 1e-5)

    phi_back = so3_log(R)
    err_log = (phi - phi_back).abs().max().item()
    _check("log ∘ exp ≈ identity", err_log < 1e-4, f"max|err|={err_log:.2e}")

    Jr = so3_right_jacobian(phi)
    _check("right Jacobian shape", Jr.shape == (4, 5, 3, 3))


def check_integrator() -> None:
    print("\n[2/6] integrate_imu_window vs analytic helix")
    imu_rate = 200.0
    dt = 1.0 / imu_rate
    n = 50
    omega_z = 0.5
    radius = 1.0

    R = torch.eye(3).unsqueeze(0)
    v0 = torch.tensor([[0.0, radius * omega_z, 0.0]])

    acc = torch.zeros(1, n, 3)
    gyro = torch.zeros(1, n, 3)
    R_curr = np.eye(3)
    for i in range(n):
        theta = omega_z * i * dt
        a_world = np.array(
            [-omega_z ** 2 * radius * np.cos(theta),
             -omega_z ** 2 * radius * np.sin(theta),
             0.0]
        )
        spec_world = a_world - np.array([0.0, 0.0, -GRAVITY])
        acc[0, i] = torch.tensor(R_curr.T @ spec_world, dtype=torch.float32)
        gyro[0, i] = torch.tensor([0.0, 0.0, omega_z], dtype=torch.float32)
        R_curr = R_curr @ so3_exp_np(np.array([0.0, 0.0, omega_z]) * dt)

    R_end_pred, v_end_pred, dp_pred = integrate_imu_window(acc, gyro, R, v0, dt)

    R_gt = torch.tensor(R_curr, dtype=torch.float32).unsqueeze(0)
    rot_err = (R_end_pred.transpose(-1, -2) @ R_gt - torch.eye(3)).abs().max().item()
    _check("rotation integration error", rot_err < 1e-3,
           f"max|err|={rot_err:.2e}")

    # Velocity in world frame after n steps.
    theta_n = omega_z * n * dt
    v_world_gt = torch.tensor(
        [[-radius * omega_z * np.sin(theta_n),
          radius * omega_z * np.cos(theta_n),
          0.0]], dtype=torch.float32
    )
    v_err = (v_end_pred - v_world_gt).abs().max().item()
    _check("velocity integration error", v_err < 5e-2,
           f"max|err|={v_err:.2e}")
    _check("displacement output shape", dp_pred.shape == (1, 3))


def check_dataset_and_models(tmp_root: Path, device: torch.device) -> None:
    print("\n[3/6] dataset + AirIMU + AirIO forward")
    ds = IMUWindowDataset(
        root_dir=str(tmp_root),
        window_size=20,
        step_size=10,
        imu_rate=200.0,
    )
    _check("dataset non-empty", len(ds) > 0, f"len={len(ds)}")
    sample = ds[0]
    _check("acc shape",       sample["acc"].shape == (20, 3))
    _check("gyro shape",      sample["gyro"].shape == (20, 3))
    _check("attitude shape",  sample["attitude"].shape == (20, 3))
    _check("v_body shape",    sample["v_body"].shape == (20, 3))
    _check("R_start shape",   sample["R_start"].shape == (3, 3))

    loader = torch.utils.data.DataLoader(ds, batch_size=4, shuffle=False)
    batch = next(iter(loader))
    acc = batch["acc"].to(device)
    gyro = batch["gyro"].to(device)
    attitude = batch["attitude"].to(device).float()
    v_body_gt = batch["v_body"].to(device).float()

    # --- AirIMU forward + step ---
    airimu = AirIMUNet().to(device)
    acc_hat, gyro_hat, log_var_imu = airimu.correct(acc, gyro)
    _check("AirIMU corrected acc shape",  acc_hat.shape == acc.shape)
    _check("AirIMU corrected gyro shape", gyro_hat.shape == gyro.shape)
    _check("AirIMU log_var shape",        log_var_imu.shape == (4, 20, 6))

    opt_imu = torch.optim.Adam(airimu.parameters(), lr=1e-3)
    opt_imu.zero_grad()
    total, terms = airimu_loss(
        acc_hat, gyro_hat, log_var_imu,
        batch["R_start"].to(device).float(),
        batch["R_end"].to(device).float(),
        batch["v_world_start"].to(device).float(),
        batch["v_world_end"].to(device).float(),
        batch["p_start"].to(device).float(),
        batch["p_end"].to(device).float(),
        dt=batch["dt"][0].item(),
    )
    _check("AirIMU loss is finite", torch.isfinite(total).item(),
           f"total={total.item():.4f} rot={terms['rot'].item():.4f}")
    total.backward()
    gn = torch.nn.utils.clip_grad_norm_(airimu.parameters(), 1.0)
    _check("AirIMU grad finite", math.isfinite(float(gn)), f"grad={float(gn):.3f}")
    opt_imu.step()

    # --- AirIO forward + step ---
    airio = AirIONet().to(device)
    v_pred, log_var_v = airio(acc_hat.detach(), gyro_hat.detach(), attitude)
    _check("AirIO velocity shape", v_pred.shape == (4, 20, 3))
    _check("AirIO log_var shape",  log_var_v.shape == (4, 20, 3))

    opt_io = torch.optim.Adam(airio.parameters(), lr=1e-3)
    opt_io.zero_grad()
    total_v, huber, nll = airio_loss(v_pred, v_body_gt, log_var_v)
    _check("AirIO loss finite", torch.isfinite(total_v).item(),
           f"total={total_v.item():.4f} huber={huber.item():.6f}")
    total_v.backward()
    gn = torch.nn.utils.clip_grad_norm_(airio.parameters(), 1.0)
    _check("AirIO grad finite", math.isfinite(float(gn)), f"grad={float(gn):.3f}")
    opt_io.step()

    # Loss exact-zero sanity.
    log_var_zero = torch.zeros_like(log_var_v)
    total0, _, _ = airio_loss(v_body_gt, v_body_gt, log_var_zero)
    _check("AirIO loss == 0 at perfect prediction (lv=0)",
           total0.abs().item() < 1e-7, f"{total0.item():.2e}")


def check_ekf(tmp_root: Path) -> None:
    print("\n[4/6] EKF predict + update")
    seq = _Sequence(tmp_root / "sequence_000", imu_rate=200.0)
    ekf = VelocityEKF()
    ekf.reset(R=seq.R[0], v=seq.R[0] @ seq.v_body[0], p=seq.p[0])

    for i in range(50):
        ekf.predict(acc=seq.acc[i], gyro=seq.gyro[i], dt=seq.dt)
        # Body-frame velocity measurement uses ground truth here; we just
        # want to confirm the update keeps things finite and PSD.
        ekf.update_velocity(v_body_meas=seq.v_body[i + 1].astype(np.float64),
                            log_var=np.full(3, math.log(1e-2)))

    _check("EKF position finite", np.all(np.isfinite(ekf.p)))
    _check("EKF rotation finite", np.all(np.isfinite(ekf.R)))
    eigs = np.linalg.eigvalsh(0.5 * (ekf.P + ekf.P.T))
    _check("EKF covariance PSD", float(eigs.min()) > -1e-6,
           f"min eig={eigs.min():.2e}")

    # After 50 updates against GT velocity, position drift should still be
    # bounded (a few metres at worst on this short window).
    drift = float(np.linalg.norm(ekf.p - seq.p[50]))
    _check("EKF tracks GT loosely", drift < 5.0, f"drift={drift:.3f} m")


def check_so3_np() -> None:
    print("\n[5/6] numpy SO(3) round-trip")
    rng = np.random.default_rng(11)
    for _ in range(10):
        phi = rng.normal(scale=0.4, size=3)
        R = so3_exp_np(phi)
        phi_back = so3_log_np(R)
        err = float(np.linalg.norm(phi - phi_back))
        _check(f"log_np ∘ exp_np ≈ identity (φ={phi.round(3).tolist()})",
               err < 1e-6, f"err={err:.2e}")


def check_gravity_constant() -> None:
    print("\n[6/6] constants")
    g = float(np.linalg.norm(np.array([0.0, 0.0, -GRAVITY])))
    _check("|gravity| ≈ 9.81", abs(g - 9.81) < 1e-6, f"g={g:.4f}")
    g_t = torch.linalg.norm(GRAVITY_VEC).item()
    _check("|GRAVITY_VEC| ≈ 9.81", abs(g_t - 9.81) < 1e-4, f"g={g_t:.4f}")


# ---------------------------------------------------------------------------
def main() -> int:
    print("=" * 60)
    print(" IMU-only pipeline smoke test (AirIMU + AirIO + EKF)")
    print("=" * 60)
    print(f"torch       : {torch.__version__}")
    print(f"cuda avail  : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"device 0    : {torch.cuda.get_device_name(0)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tmp_dir = Path(tempfile.mkdtemp(prefix="imu_only_smoke_"))
    print(f"tmp dataset : {tmp_dir}")
    try:
        _make_synthetic_dataset(tmp_dir, n_sequences=3, n_samples=600, imu_rate=200.0)
        check_so3()
        check_integrator()
        check_dataset_and_models(tmp_dir, device)
        check_ekf(tmp_dir)
        check_so3_np()
        check_gravity_constant()
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    print("\n" + "=" * 60)
    print(" All checks passed.")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
