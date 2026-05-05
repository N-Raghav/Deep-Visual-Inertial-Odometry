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


def integrate_relative_poses(
    R_rel: torch.Tensor, t_rel: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Differentiably integrate relative poses into world-frame poses.

    Args:
        R_rel: ``[B, T, 3, 3]`` relative rotations.
        t_rel: ``[B, T, 3]`` relative translations.

    Returns:
        ``(p_world, R_world)`` both starting from identity, shape ``[B, T+1, ...]``.
    """
    B = t_rel.shape[0]
    device, dtype = t_rel.device, t_rel.dtype
    R_cur = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).expand(B, -1, -1)
    p_cur = torch.zeros(B, 3, device=device, dtype=dtype)
    p_list, R_list = [p_cur], [R_cur]
    for i in range(t_rel.shape[1]):
        p_cur = p_cur + (R_cur @ t_rel[:, i].unsqueeze(-1)).squeeze(-1)
        R_cur = R_cur @ R_rel[:, i]
        p_list.append(p_cur)
        R_list.append(R_cur)
    return torch.stack(p_list, dim=1), torch.stack(R_list, dim=1)


def trajectory_loss(
    R_pred: torch.Tensor,
    t_pred: torch.Tensor,
    R_gt: torch.Tensor,
    t_gt: torch.Tensor,
    lambda_rot: float = 10.0,
) -> torch.Tensor:
    """Trajectory-level loss: penalize drift in the integrated mini-trajectory.

    Integrates T relative poses into world positions and rotations, then
    supervises both (smooth-L1 on positions, geodesic on accumulated rotations).
    This penalises systematic per-step bias that per-pair loss cannot see.

    Args:
        R_pred, R_gt: ``[B, T, 3, 3]``.
        t_pred, t_gt: ``[B, T, 3]``.
        lambda_rot: weight on the rotation term within this loss.
    """
    p_pred, R_pred_w = integrate_relative_poses(R_pred, t_pred)
    p_gt, R_gt_w = integrate_relative_poses(R_gt, t_gt)

    pos_loss = F.smooth_l1_loss(p_pred[:, 1:], p_gt[:, 1:])

    B, T1 = R_pred_w.shape[:2]
    T = T1 - 1
    rot_loss = geodesic_loss_stable(
        R_pred_w[:, 1:].reshape(B * T, 3, 3),
        R_gt_w[:, 1:].reshape(B * T, 3, 3),
    )
    return pos_loss + lambda_rot * rot_loss


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
