"""
Loss functions for Branch A.

Provides:
    - geodesic_loss_stable: numerically stable angular distance on SO(3).
    - branch_a_loss:        weighted sum of smooth-L1 translation loss and
                            geodesic rotation loss.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def geodesic_loss_stable(R_pred: torch.Tensor, R_gt: torch.Tensor) -> torch.Tensor:
    """Mean geodesic distance between predicted and ground-truth rotations.

    Computes the rotation angle of ``R_pred^T @ R_gt`` using the
    ``atan2(||skew||, (trace - 1) / 2)`` formulation, which is stable at
    both ``angle = 0`` and ``angle = pi`` (the ``arccos`` formulation has
    infinite gradient at those points).

    Args:
        R_pred: Predicted rotation matrices, shape ``[..., 3, 3]``.
        R_gt: Ground-truth rotation matrices, shape ``[..., 3, 3]``.

    Returns:
        Scalar mean geodesic distance in radians.
    """
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
    angle = torch.atan2(skew_norm, cos_angle)
    return angle.mean()


def branch_a_loss(
    trans_pred: torch.Tensor,
    trans_gt: torch.Tensor,
    R_pred: torch.Tensor,
    R_gt: torch.Tensor,
    lambda_rot: float = 100.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Combined translation + rotation loss for Branch A.

    Args:
        trans_pred: Predicted translations, shape ``[..., 3]``.
        trans_gt: Ground-truth translations, shape ``[..., 3]``.
        R_pred: Predicted rotation matrices, shape ``[..., 3, 3]``.
        R_gt: Ground-truth rotation matrices, shape ``[..., 3, 3]``.
        lambda_rot: Weight on the rotation term. Translation is roughly
            in metres while rotation is in radians, so a multiplier of
            ~100 brings the two onto a comparable scale for typical UAV
            motion magnitudes.

    Returns:
        Tuple ``(total_loss, trans_loss, rot_loss)`` of scalar tensors,
        with ``trans_loss`` and ``rot_loss`` returned for logging.
    """
    trans_loss = F.smooth_l1_loss(trans_pred, trans_gt)
    rot_loss = geodesic_loss_stable(R_pred, R_gt)
    total = trans_loss + lambda_rot * rot_loss
    return total, trans_loss, rot_loss
