"""
Utility functions for the IMU-only branch (AirIMU + AirIO + EKF).

Implements:
    - SO(3) operations: hat / vee / exp / log (numerically stable, batched)
    - Right Jacobian of SO(3) (used inside the EKF)
    - Conversion of pose sequences into per-step body-frame velocities and
      attitude logarithms (training labels for AirIO)
    - A simple IMU pre-integrator used as a baseline / inside AirIMU's loss

All functions accept arbitrary leading batch dimensions via ``...`` where
possible.
"""

from __future__ import annotations

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
GRAVITY = 9.81
# World gravity vector. Convention: +Z is up, gravity acts in -Z, so the
# accelerometer at rest reads +9.81 along its body-Z axis when level.
GRAVITY_VEC = torch.tensor([0.0, 0.0, -GRAVITY])


# ---------------------------------------------------------------------------
# SO(3) helpers (PyTorch)
# ---------------------------------------------------------------------------
def so3_hat(v: torch.Tensor) -> torch.Tensor:
    """Vector to skew-symmetric matrix.

    Args:
        v: ``[..., 3]``.
    Returns:
        ``[..., 3, 3]`` skew-symmetric matrix.
    """
    z = torch.zeros_like(v[..., 0])
    M = torch.stack(
        [
            torch.stack([z, -v[..., 2], v[..., 1]], dim=-1),
            torch.stack([v[..., 2], z, -v[..., 0]], dim=-1),
            torch.stack([-v[..., 1], v[..., 0], z], dim=-1),
        ],
        dim=-2,
    )
    return M


def so3_vee(M: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`so3_hat`."""
    return torch.stack([M[..., 2, 1], M[..., 0, 2], M[..., 1, 0]], dim=-1)


def so3_exp(phi: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Exponential map ``so(3) -> SO(3)``.

    Uses the closed-form Rodrigues formula with a small-angle Taylor
    expansion for numerical stability around ``||phi|| = 0``.

    Args:
        phi: ``[..., 3]`` rotation vector (axis * angle).
    Returns:
        ``[..., 3, 3]`` rotation matrix.
    """
    theta = torch.linalg.norm(phi, dim=-1, keepdim=True).unsqueeze(-1)  # [...,1,1]
    K = so3_hat(phi)
    K2 = K @ K
    eye = torch.eye(3, device=phi.device, dtype=phi.dtype).expand(K.shape)

    small = theta < 1e-4
    # First-order series for small angles: A = 1 - θ²/6, B = 1/2 - θ²/24.
    A_small = 1.0 - theta ** 2 / 6.0
    B_small = 0.5 - theta ** 2 / 24.0
    A_full = torch.sin(theta.clamp(min=eps)) / theta.clamp(min=eps)
    B_full = (1.0 - torch.cos(theta.clamp(min=eps))) / (theta.clamp(min=eps) ** 2)
    A = torch.where(small, A_small, A_full)
    B = torch.where(small, B_small, B_full)

    return eye + A * K + B * K2


def so3_log(R: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Logarithm map ``SO(3) -> so(3)``.

    Args:
        R: ``[..., 3, 3]`` rotation matrices.
    Returns:
        ``[..., 3]`` rotation vector (axis * angle, range ``[0, pi]``).
    """
    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    cos_theta = torch.clamp((trace - 1.0) / 2.0, -1.0 + eps, 1.0 - eps)
    theta = torch.acos(cos_theta)  # [...]

    # skew_part: vee((R - R^T) / 2) — but the proper scaling is θ / (2 sin θ).
    skew_vec = 0.5 * torch.stack(
        [
            R[..., 2, 1] - R[..., 1, 2],
            R[..., 0, 2] - R[..., 2, 0],
            R[..., 1, 0] - R[..., 0, 1],
        ],
        dim=-1,
    )
    sin_theta = torch.sin(theta)

    small = theta < 1e-4
    # Small-angle: scale = 1 + θ²/6 (Taylor of θ / sin θ).
    scale_small = 1.0 + theta ** 2 / 6.0
    scale_full = theta / sin_theta.clamp(min=eps)
    scale = torch.where(small, scale_small, scale_full).unsqueeze(-1)
    return skew_vec * scale


def so3_right_jacobian(phi: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Right Jacobian of SO(3) at ``phi``.

    ``J_r(phi)`` linearises ``Exp(phi + δphi) ≈ Exp(phi) Exp(J_r δphi)``.
    """
    theta = torch.linalg.norm(phi, dim=-1, keepdim=True).unsqueeze(-1)
    K = so3_hat(phi)
    eye = torch.eye(3, device=phi.device, dtype=phi.dtype).expand(K.shape)

    small = theta < 1e-4
    A_small = 0.5 - theta ** 2 / 24.0
    B_small = (1.0 / 6.0) - theta ** 2 / 120.0
    A_full = (1.0 - torch.cos(theta.clamp(min=eps))) / (theta.clamp(min=eps) ** 2)
    B_full = (theta - torch.sin(theta.clamp(min=eps))) / (theta.clamp(min=eps) ** 3)
    A = torch.where(small, A_small, A_full)
    B = torch.where(small, B_small, B_full)
    return eye - A * K + B * (K @ K)


# ---------------------------------------------------------------------------
# Numpy versions (used by EKF in evaluation, pose-loading code)
# ---------------------------------------------------------------------------
def so3_hat_np(v: np.ndarray) -> np.ndarray:
    return np.array(
        [[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]], dtype=v.dtype
    )


def so3_exp_np(phi: np.ndarray) -> np.ndarray:
    theta = float(np.linalg.norm(phi))
    K = so3_hat_np(phi)
    if theta < 1e-8:
        return np.eye(3) + K + 0.5 * K @ K
    A = np.sin(theta) / theta
    B = (1.0 - np.cos(theta)) / (theta ** 2)
    return np.eye(3) + A * K + B * (K @ K)


def so3_log_np(R: np.ndarray) -> np.ndarray:
    cos_theta = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    theta = float(np.arccos(cos_theta))
    if theta < 1e-7:
        return 0.5 * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return (theta / (2.0 * np.sin(theta))) * np.array(
        [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]
    )


# ---------------------------------------------------------------------------
# Pose-sequence helpers
# ---------------------------------------------------------------------------
def body_frame_velocity_from_poses(
    R: np.ndarray, p: np.ndarray, dt: float | np.ndarray
) -> np.ndarray:
    """Compute per-sample body-frame velocity ``ᴮv_i`` from a pose sequence.

    Uses a centred finite difference where possible; the first and last
    samples fall back to forward / backward differences.

    Args:
        R: ``(N, 3, 3)`` per-sample rotations (body -> world).
        p: ``(N, 3)`` per-sample positions in world frame.
        dt: scalar timestep (seconds) or ``(N - 1,)`` array of intervals.

    Returns:
        ``(N, 3)`` body-frame velocity ``ᴮv_i = R_iᵀ ᴳv_i`` per sample.
    """
    n = p.shape[0]
    v_world = np.zeros_like(p)
    if np.isscalar(dt):
        dt_arr = np.full(n - 1, float(dt))
    else:
        dt_arr = np.asarray(dt, dtype=np.float64)

    # Forward diff for sample 0, centred for the interior, backward for last.
    v_world[0] = (p[1] - p[0]) / dt_arr[0]
    if n > 2:
        v_world[1:-1] = (p[2:] - p[:-2]) / (dt_arr[:-1] + dt_arr[1:])[:, None]
    v_world[-1] = (p[-1] - p[-2]) / dt_arr[-1]

    v_body = np.einsum("nij,nj->ni", np.transpose(R, (0, 2, 1)), v_world)
    return v_body


def attitude_logmap_from_poses(R: np.ndarray) -> np.ndarray:
    """Per-sample ``ξ = log_SO(3)(R) ∈ ℝ³`` for a pose sequence."""
    out = np.zeros((R.shape[0], 3), dtype=np.float64)
    for i in range(R.shape[0]):
        out[i] = so3_log_np(R[i])
    return out


# ---------------------------------------------------------------------------
# Differentiable IMU pre-integrator (used by AirIMU's training loss)
# ---------------------------------------------------------------------------
def integrate_imu_window(
    acc: torch.Tensor,
    gyro: torch.Tensor,
    R0: torch.Tensor,
    v0: torch.Tensor,
    dt: torch.Tensor | float,
    gravity: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Forward-Euler integration of an IMU window.

    All inputs are expected on the same device. Accelerations are assumed
    to be specific-force measurements (i.e. they include the gravity
    component) expressed in the body frame.

    Args:
        acc: ``[B, W, 3]`` body-frame accelerometer measurements.
        gyro: ``[B, W, 3]`` body-frame gyroscope measurements.
        R0: ``[B, 3, 3]`` rotation at the window start (body -> world).
        v0: ``[B, 3]`` world-frame velocity at the window start.
        dt: scalar or ``[B, W]`` per-step timestep.
        gravity: optional ``[3]`` world-frame gravity vector. Defaults
            to ``[0, 0, -9.81]``.

    Returns:
        Tuple ``(R_end, v_end, p_delta)`` where:
            - ``R_end``  ``[B, 3, 3]`` rotation at window end,
            - ``v_end``  ``[B, 3]`` world-frame velocity at window end,
            - ``p_delta`` ``[B, 3]`` displacement integrated over the window
              (i.e. ``p_end - p_start``).
    """
    if gravity is None:
        gravity = GRAVITY_VEC.to(acc.device, acc.dtype)

    B, W, _ = acc.shape
    if not torch.is_tensor(dt):
        dt = torch.full((B, W), float(dt), device=acc.device, dtype=acc.dtype)
    elif dt.dim() == 0:
        dt = dt.expand(B, W)

    R = R0
    v = v0
    p = torch.zeros_like(v0)

    for k in range(W):
        # Mid-point integration on rotation: R update first.
        omega_dt = gyro[:, k] * dt[:, k:k + 1]
        dR = so3_exp(omega_dt)
        R_next = R @ dR

        # Global-frame motion accel. Use rotation at start of step.
        a_world = (R @ acc[:, k].unsqueeze(-1)).squeeze(-1) + gravity
        v_next = v + a_world * dt[:, k:k + 1]
        p = p + v * dt[:, k:k + 1] + 0.5 * a_world * dt[:, k:k + 1] ** 2

        R = R_next
        v = v_next

    return R, v, p
