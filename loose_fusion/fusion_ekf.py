"""
Loose-fusion EKF: extends the AirIO velocity-measurement EKF with a
vision update step.

The base filter (``imu_only/ekf.py``) already does:

    - prediction from corrected IMU + AirIMU log-variance,
    - body-frame velocity update from AirIO (Eq. 10-12 of the AirIO paper).

This subclass adds a third update fed by the vision branch's relative
pose. Loose coupling means the two networks are *independent*: each
emits its own measurement and the EKF reconciles them — no shared
features, no joint training.

Vision update model:

    The vision branch produces a relative pose ``(ΔR_vis, Δt_vis)``
    between two consecutive frames. Convert it to a body-frame velocity
    measurement at frame ``t``:

        ᴮv_vis = Δt_vis / Δt_frame

    This makes vision a *second sensor of body-frame velocity* with a
    different rate and a different uncertainty profile from AirIO; the
    EKF update model is identical to the AirIO update.

    Optionally, the rotation increment ``ΔR_vis`` is also used as a
    measurement of the EKF's predicted rotation increment between
    frames. The innovation lives in ``so(3)`` and the Jacobian is the
    identity on ``δξ`` (small-angle approximation).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Make sibling branches importable.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from imu_only.ekf import VelocityEKF, _skew  # noqa: E402
from imu_only.utils import so3_exp_np, so3_log_np  # noqa: E402


class FusionEKF(VelocityEKF):
    """Velocity-measurement EKF with an additional vision update step."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Snapshot of the EKF rotation at the previous vision frame, used
        # to compute the predicted rotation increment.
        self._R_prev_vis: np.ndarray | None = None

    def reset(self, *args, **kwargs) -> None:
        super().reset(*args, **kwargs)
        self._R_prev_vis = self.R.copy()

    # ------------------------------------------------------------------
    def update_vision_velocity(
        self,
        delta_t_vis: np.ndarray,
        dt_frame: float,
        sigma: float = 0.05,
    ) -> None:
        """Body-frame-velocity update fed by the vision branch.

        Args:
            delta_t_vis: ``(3,)`` predicted relative translation in the
                previous frame's body frame (output of BranchA).
            dt_frame: time interval between the two vision frames.
            sigma: standard deviation (m/s) of the vision velocity noise
                — typically larger than AirIO's because vision operates
                at a lower rate and is more affected by noise / texture.
        """
        v_body_meas = np.asarray(delta_t_vis, dtype=np.float64) / float(dt_frame)
        log_var = np.full(3, 2.0 * np.log(max(sigma, 1e-6)))
        self.update_velocity(v_body_meas=v_body_meas, log_var=log_var)

    # ------------------------------------------------------------------
    def update_vision_rotation(
        self,
        delta_R_vis: np.ndarray,
        sigma_deg: float = 0.5,
    ) -> None:
        """Rotation-increment update fed by the vision branch.

        Innovation is the small-angle log of the residual rotation
        between the EKF-predicted rotation increment (``R_prev_visᵀ R``)
        and the vision-predicted increment ``ΔR_vis``.

        Args:
            delta_R_vis: ``(3, 3)`` predicted relative rotation between
                the previous and current vision frames.
            sigma_deg: standard deviation of the rotation noise, in
                degrees. Default 0.5° is a reasonable starting value for
                a learned vision branch.
        """
        if self._R_prev_vis is None:
            self._R_prev_vis = self.R.copy()
            return

        R_pred_inc = self._R_prev_vis.T @ self.R                # current EKF increment
        R_meas_inc = np.asarray(delta_R_vis, dtype=np.float64)  # vision increment
        innovation = so3_log_np(R_pred_inc.T @ R_meas_inc)      # ∈ so(3)

        # Linearised model: the increment depends on the *current* δξ
        # (the previous one is "frozen" in _R_prev_vis). Jacobian on
        # δξ is the identity to first order.
        H = np.zeros((3, 15))
        H[:, 0:3] = np.eye(3)

        sigma_rad = np.radians(sigma_deg)
        R_meas_cov = np.eye(3) * (sigma_rad ** 2)

        S = H @ self.P @ H.T + R_meas_cov
        K = self.P @ H.T @ np.linalg.inv(S)
        delta = K @ innovation

        d_xi = delta[0:3]
        d_v = delta[3:6]
        d_p = delta[6:9]
        d_ba = delta[9:12]
        d_bg = delta[12:15]

        self.R = self.R @ so3_exp_np(d_xi)
        self.v = self.v + d_v
        self.p = self.p + d_p
        self.b_a = self.b_a + d_ba
        self.b_g = self.b_g + d_bg

        I_KH = np.eye(15) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R_meas_cov @ K.T
        self.P = 0.5 * (self.P + self.P.T)

        # Update keyframe snapshot.
        self._R_prev_vis = self.R.copy()
