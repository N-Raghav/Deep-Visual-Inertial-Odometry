"""
Utility functions for Branch A.

Contains:
    - gram_schmidt: 6D rotation -> 3x3 rotation matrix (Zhou et al. CVPR 2019)
    - pose_to_matrix: build 4x4 homogeneous transform from R and t
    - matrix_to_pose: split a 4x4 transform into R and t
    - umeyama_alignment: similarity transform alignment for trajectory eval
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def gram_schmidt(r6: torch.Tensor) -> torch.Tensor:
    """Recover a 3x3 rotation matrix from a 6D continuous representation.

    Implements the Gram-Schmidt orthogonalization of Zhou et al.
    (CVPR 2019, "On the Continuity of Rotation Representations in
    Neural Networks").

    Supports arbitrary leading batch dimensions via the ``...`` notation,
    so both ``[B, 6]`` and ``[B, T, 6]`` shapes are accepted.

    Args:
        r6: Tensor of shape ``[..., 6]`` containing the raw network output.
            ``r6[..., :3]`` are interpreted as the (unnormalized) first
            column of the rotation matrix and ``r6[..., 3:]`` as the
            (unnormalized) second column.

    Returns:
        Rotation matrix of shape ``[..., 3, 3]``. The columns of the
        returned matrix form a right-handed orthonormal basis.
    """
    a1 = r6[..., :3]
    a2 = r6[..., 3:]

    b1 = F.normalize(a1, dim=-1, eps=1e-8)
    # Project a2 onto b1 and subtract to enforce orthogonality.
    dot = (b1 * a2).sum(dim=-1, keepdim=True)
    b2 = F.normalize(a2 - dot * b1, dim=-1, eps=1e-8)
    b3 = torch.cross(b1, b2, dim=-1)

    # Stack as columns of the rotation matrix: R = [b1 | b2 | b3].
    R = torch.stack([b1, b2, b3], dim=-1)
    return R


def pose_to_matrix(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Combine a rotation matrix and translation vector into a 4x4 transform.

    Args:
        R: Rotation of shape ``(3, 3)``.
        t: Translation of shape ``(3,)`` or ``(3, 1)``.

    Returns:
        4x4 homogeneous transform of shape ``(4, 4)``.
    """
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t).reshape(3)
    return T


def matrix_to_pose(T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split a 4x4 homogeneous transform into ``(R, t)``.

    Args:
        T: 4x4 homogeneous transform.

    Returns:
        Tuple ``(R, t)`` with shapes ``(3, 3)`` and ``(3,)``.
    """
    R = T[:3, :3]
    t = T[:3, 3]
    return R, t


def umeyama_alignment(
    pred: np.ndarray, gt: np.ndarray, with_scale: bool = True
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """Compute the similarity transform aligning ``pred`` to ``gt``.

    Implements the closed-form least-squares solution of
    Umeyama, "Least-squares estimation of transformation parameters
    between two point patterns," IEEE TPAMI 1991.

    Args:
        pred: Predicted points, shape ``(N, 3)``.
        gt: Ground-truth points, shape ``(N, 3)``.
        with_scale: If ``True``, estimate an isotropic scale (similarity
            transform). If ``False``, only estimate rotation and
            translation (rigid transform).

    Returns:
        Tuple ``(R, t, s, aligned)`` where:
            - ``R`` is the ``(3, 3)`` rotation,
            - ``t`` is the ``(3,)`` translation,
            - ``s`` is the scalar scale (1.0 if ``with_scale=False``),
            - ``aligned`` is ``pred`` after applying ``s * R @ p + t``.
    """
    assert pred.shape == gt.shape and pred.shape[1] == 3, (
        f"shape mismatch: {pred.shape} vs {gt.shape}"
    )
    n = pred.shape[0]

    mu_pred = pred.mean(axis=0)
    mu_gt = gt.mean(axis=0)

    pred_c = pred - mu_pred
    gt_c = gt - mu_gt

    # Cross-covariance.
    cov = (gt_c.T @ pred_c) / n
    U, D, Vt = np.linalg.svd(cov)

    # Reflection-handling: enforce det(R) = +1.
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1.0

    R = U @ S @ Vt

    if with_scale:
        var_pred = (pred_c ** 2).sum() / n
        s = float(np.trace(np.diag(D) @ S) / var_pred) if var_pred > 0 else 1.0
    else:
        s = 1.0

    t = mu_gt - s * R @ mu_pred
    aligned = (s * (R @ pred.T)).T + t
    return R, t, s, aligned


def integrate_trajectory(
    rel_R: np.ndarray, rel_t: np.ndarray, T0: np.ndarray | None = None
) -> np.ndarray:
    """Dead-reckon a sequence of relative poses into world poses.

    Args:
        rel_R: Relative rotations, shape ``(N, 3, 3)``.
        rel_t: Relative translations, shape ``(N, 3)``.
        T0: Optional starting world pose ``(4, 4)``. Defaults to identity.

    Returns:
        World poses of shape ``(N + 1, 4, 4)`` starting from ``T0``.
    """
    if T0 is None:
        T0 = np.eye(4, dtype=np.float64)
    poses = [T0.copy()]
    for i in range(rel_R.shape[0]):
        T_rel = pose_to_matrix(rel_R[i], rel_t[i])
        poses.append(poses[-1] @ T_rel)
    return np.stack(poses, axis=0)


def rotation_geodesic_deg(R_pred: np.ndarray, R_gt: np.ndarray) -> np.ndarray:
    """Per-frame geodesic rotation error in degrees.

    Args:
        R_pred: Predicted rotations, shape ``(N, 3, 3)``.
        R_gt: Ground-truth rotations, shape ``(N, 3, 3)``.

    Returns:
        Array of shape ``(N,)`` with geodesic distances in degrees.
    """
    R_diff = np.matmul(np.transpose(R_pred, (0, 2, 1)), R_gt)
    trace = np.trace(R_diff, axis1=1, axis2=2)
    skew = np.stack(
        [
            R_diff[:, 2, 1] - R_diff[:, 1, 2],
            R_diff[:, 0, 2] - R_diff[:, 2, 0],
            R_diff[:, 1, 0] - R_diff[:, 0, 1],
        ],
        axis=-1,
    )
    skew_norm = np.linalg.norm(skew, axis=-1) / 2.0
    cos_angle = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
    angle_rad = np.arctan2(skew_norm, cos_angle)
    return np.degrees(angle_rad)
