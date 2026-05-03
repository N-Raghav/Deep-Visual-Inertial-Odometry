"""
Pose loss shared by the tight-fusion training scripts.

Two terms, the same as in [vision_only/loss.py](../vision_only/loss.py):

    - smooth-L1 on the 3D relative translation,
    - geodesic angle on SO(3) for the relative rotation, computed via the
      atan2 formulation for numerical stability around 0 and π.

The combined loss returns the auxiliary terms for logging.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def geodesic_loss_stable(R_pred: torch.Tensor, R_gt: torch.Tensor) -> torch.Tensor:
    """Mean geodesic distance between predicted and ground-truth rotations."""
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


def pose_loss(
    trans_pred: torch.Tensor,
    trans_gt: torch.Tensor,
    R_pred: torch.Tensor,
    R_gt: torch.Tensor,
    lambda_rot: float = 100.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Combined translation + rotation loss for tight fusion training.

    Args:
        trans_pred / trans_gt: ``[..., 3]`` predicted / ground-truth
            translations.
        R_pred / R_gt: ``[..., 3, 3]`` rotation matrices.
        lambda_rot: weight on the rotation term, balancing meter-scale
            translations against radian-scale rotations.

    Returns:
        ``(total, trans_loss, rot_loss)`` — the auxiliary terms are
        returned so training scripts can log them separately.
    """
    trans_loss = F.smooth_l1_loss(trans_pred, trans_gt)
    rot_loss = geodesic_loss_stable(R_pred, R_gt)
    total = trans_loss + lambda_rot * rot_loss
    return total, trans_loss, rot_loss
