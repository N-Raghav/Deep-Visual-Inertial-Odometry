"""
Trajectory metrics shared by the three fusion ``evaluate.py`` scripts.

All inputs are numpy arrays and follow the AirIO paper's conventions:

    - **ATE (Eq. 13)**: RMSE of per-sample world-frame translation error.
      In the paper this is computed without alignment because the start
      pose is known. We support both ``aligned=True`` (Umeyama similarity
      alignment, useful when the trajectory starts from the origin and
      drifts) and ``aligned=False`` (raw RMSE).

    - **RTE (Eq. 14)**: relative translation error at a fixed time
      interval (paper uses 5 s). Local frame: each ground-truth interval
      is rotated into the body frame at its start so longer translations
      do not dominate.

    - **Mean rotation error**: geodesic angle in degrees, mean across all
      samples.
"""

from __future__ import annotations

import numpy as np


def umeyama_alignment(
    pred: np.ndarray, gt: np.ndarray, with_scale: bool = True
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """Closed-form similarity transform aligning ``pred`` to ``gt``."""
    assert pred.shape == gt.shape and pred.shape[1] == 3
    n = pred.shape[0]
    mu_p = pred.mean(axis=0)
    mu_g = gt.mean(axis=0)
    pc = pred - mu_p
    gc = gt - mu_g
    cov = (gc.T @ pc) / n
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1.0
    R = U @ S @ Vt
    if with_scale:
        var_p = (pc ** 2).sum() / n
        s = float(np.trace(np.diag(D) @ S) / var_p) if var_p > 0 else 1.0
    else:
        s = 1.0
    t = mu_g - s * R @ mu_p
    aligned = (s * (R @ pred.T)).T + t
    return R, t, s, aligned


def trajectory_metrics(
    pos_pred: np.ndarray,
    pos_gt: np.ndarray,
    R_pred: np.ndarray,
    R_gt: np.ndarray,
    rte_step: int,
    aligned: bool = False,
) -> dict[str, float | np.ndarray]:
    """Compute ATE, RTE, mean rotation error and per-sample error series.

    Args:
        pos_pred / pos_gt: ``(N, 3)`` world-frame positions.
        R_pred / R_gt: ``(N, 3, 3)`` world-frame rotations.
        rte_step: number of samples corresponding to the RTE interval
            (e.g. ``round(imu_rate * 5.0)`` for the paper's 5-second
            interval).
        aligned: if ``True``, Umeyama-align ``pos_pred`` to ``pos_gt``
            before computing ATE.

    Returns:
        dict with keys ``ate``, ``rte``, ``rot_deg``, ``per_frame_trans``,
        ``per_frame_rot_deg``.
    """
    if aligned:
        _, _, _, pos_pred_eval = umeyama_alignment(pos_pred, pos_gt, with_scale=True)
    else:
        pos_pred_eval = pos_pred

    per_frame_trans = np.linalg.norm(pos_pred_eval - pos_gt, axis=1)
    ate = float(np.sqrt(np.mean(per_frame_trans ** 2)))

    n = pos_pred.shape[0]
    rte_residuals = []
    for i in range(0, n - rte_step):
        gt_d_world = pos_gt[i + rte_step] - pos_gt[i]
        gt_d_local = R_gt[i].T @ gt_d_world
        pred_d_world = pos_pred[i + rte_step] - pos_pred[i]
        pred_d_local = R_pred[i].T @ pred_d_world
        rte_residuals.append(float(np.linalg.norm(pred_d_local - gt_d_local)))
    rte = float(np.sqrt(np.mean(np.square(rte_residuals)))) if rte_residuals else 0.0

    rot_per_frame_deg = np.zeros(n)
    for i in range(n):
        R_diff = R_pred[i].T @ R_gt[i]
        cos_a = np.clip((np.trace(R_diff) - 1.0) / 2.0, -1.0, 1.0)
        rot_per_frame_deg[i] = float(np.degrees(np.arccos(cos_a)))
    rot_deg = float(rot_per_frame_deg.mean())

    return {
        "ate": ate,
        "rte": rte,
        "rot_deg": rot_deg,
        "per_frame_trans": per_frame_trans,
        "per_frame_rot_deg": rot_per_frame_deg,
    }
