"""
Loss functions for the IMU-only branch.

Provides:
    - ``gaussian_nll``: per-element Gaussian negative log-likelihood with
      a learned diagonal log-variance (Eq. 5 of the AirIO paper).
    - ``airio_loss``: Huber-on-velocity + λ * Gaussian NLL (Eq. 3).
    - ``airimu_loss``: rotation + velocity + position residuals between
      the AirIMU-corrected pre-integrated window and ground truth, with
      an additional NLL on the IMU uncertainty.
    - ``geodesic_loss_stable``: shared with the vision branch — atan2
      formulation for numerical stability at angles 0 and π.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from utils import integrate_imu_window, so3_log


def geodesic_loss_stable(R_pred: torch.Tensor, R_gt: torch.Tensor) -> torch.Tensor:
    """Mean geodesic angle on SO(3), atan2 formulation."""
    R_diff = torch.matmul(R_pred.transpose(-1, -2), R_gt)
    trace = R_diff[..., 0, 0] + R_diff[..., 1, 1] + R_diff[..., 2, 2]
    skew = torch.stack(
        [
            R_diff[..., 2, 1] - R_diff[..., 1, 2],
            R_diff[..., 0, 2] - R_diff[..., 2, 0],
            R_diff[..., 1, 0] - R_diff[..., 0, 1],
        ],
        dim=-1,
    )
    skew_norm = torch.linalg.norm(skew, dim=-1) / 2.0
    cos_angle = torch.clamp((trace - 1.0) / 2.0, -1.0, 1.0)
    return torch.atan2(skew_norm, cos_angle).mean()


def gaussian_nll(
    residual: torch.Tensor, log_var: torch.Tensor
) -> torch.Tensor:
    """Diagonal-Gaussian negative log-likelihood.

    NLL per element is ``0.5 * (residual² / σ² + log σ²)``; we drop the
    constant ``0.5 * log(2π)`` term. Mean is taken over all elements.

    Args:
        residual: tensor of any shape ``[..., D]``.
        log_var: same shape as ``residual``; log of the variance.
    """
    var = torch.exp(log_var)
    return 0.5 * ((residual ** 2) / var + log_var).mean()


# ---------------------------------------------------------------------------
# AirIO loss (Eq. 3-5 of the paper)
# ---------------------------------------------------------------------------
def airio_loss(
    v_pred: torch.Tensor,
    v_gt: torch.Tensor,
    log_var: torch.Tensor,
    huber_delta: float = 0.005,
    lambda_c: float = 1e-4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Combined Huber + Gaussian-NLL loss on body-frame velocity.

    Args:
        v_pred: ``[..., 3]`` predicted body-frame velocity.
        v_gt: ``[..., 3]`` ground-truth body-frame velocity.
        log_var: ``[..., 3]`` log of the diagonal velocity variance.
        huber_delta: Huber transition parameter (paper uses 0.005).
        lambda_c: weight on the NLL term (paper uses 1e-4).

    Returns:
        Tuple ``(total, huber, nll)`` of scalars; the auxiliary terms are
        returned for logging.
    """
    huber = F.huber_loss(v_pred, v_gt, delta=huber_delta)
    nll = gaussian_nll(v_pred - v_gt, log_var)
    total = huber + lambda_c * nll
    return total, huber, nll


# ---------------------------------------------------------------------------
# AirIMU loss
# ---------------------------------------------------------------------------
def airimu_loss(
    acc_hat: torch.Tensor,
    gyro_hat: torch.Tensor,
    log_var: torch.Tensor,
    R_start: torch.Tensor,
    R_end: torch.Tensor,
    v_world_start: torch.Tensor,
    v_world_end: torch.Tensor,
    p_start: torch.Tensor,
    p_end: torch.Tensor,
    dt: torch.Tensor | float,
    lambda_rot: float = 10.0,
    lambda_vel: float = 1.0,
    lambda_pos: float = 1.0,
    lambda_nll: float = 1e-4,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Pre-integration loss for AirIMU.

    Integrates the corrected IMU window with a forward-Euler integrator
    (see :func:`utils.integrate_imu_window`) and compares the resulting
    end-of-window rotation, velocity, and displacement to ground truth.
    A Gaussian NLL on the residual encourages the network to honestly
    report its own uncertainty.

    Args:
        acc_hat:  ``[B, W, 3]`` AirIMU-corrected accel measurements.
        gyro_hat: ``[B, W, 3]`` AirIMU-corrected gyro measurements.
        log_var:  ``[B, W, 6]`` predicted log-variance per IMU sample.
        R_start, R_end:        rotations at the window endpoints, ``[B, 3, 3]``.
        v_world_start, v_world_end: world-frame velocities, ``[B, 3]``.
        p_start, p_end:        world-frame positions, ``[B, 3]``.
        dt: scalar timestep or ``[B, W]`` array.
        lambda_*: relative weights of the four terms.

    Returns:
        Tuple ``(total_loss, terms)`` where ``terms`` is a dict with
        keys ``rot``, ``vel``, ``pos``, ``nll`` for logging.
    """
    R_int, v_int, dp_int = integrate_imu_window(
        acc_hat, gyro_hat, R_start, v_world_start, dt
    )
    rot_loss = geodesic_loss_stable(R_int, R_end)
    vel_loss = F.mse_loss(v_int, v_world_end)
    pos_loss = F.mse_loss(dp_int, p_end - p_start)

    # Pseudo-residual for NLL: per-window, broadcast a single 6-dim
    # rotation+velocity error against every sample's log_var. This trains
    # the per-sample uncertainty to be at least as large as the typical
    # window-end residual.
    rot_resid = so3_log(R_int.transpose(-1, -2) @ R_end)        # [B, 3]
    vel_resid = v_int - v_world_end                              # [B, 3]
    resid = torch.cat([vel_resid, rot_resid], dim=-1)            # [B, 6]
    nll_term = gaussian_nll(
        resid.unsqueeze(1).expand_as(log_var), log_var
    )

    total = (
        lambda_rot * rot_loss
        + lambda_vel * vel_loss
        + lambda_pos * pos_loss
        + lambda_nll * nll_term
    )
    return total, {
        "rot": rot_loss.detach(),
        "vel": vel_loss.detach(),
        "pos": pos_loss.detach(),
        "nll": nll_term.detach(),
    }
