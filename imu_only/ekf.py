"""
Velocity-measurement Extended Kalman Filter (Eq. 7-12 of the AirIO paper).

State (15-dim error state):
    δX = [δξ, δᴳv, δᴳp, δb_a, δb_g] ∈ ℝ¹⁵

Nominal state:
    X = (R, ᴳv, ᴳp, b_a, b_g)

Propagation uses corrected IMU samples (``â = a + σ̂_a``,
``ŵ = ω + σ̂_g``) and an explicit gravity term in the global frame.
Process noise is the AirIMU-predicted per-frame uncertainty plus a small
constant random walk on the biases.

The measurement model is the AirIO-predicted body-frame velocity:
    h(X) = Rᵀ ᴳv = ᴮv

which yields the Jacobians (Eq. 11):
    H_{δᴳv} =  Rᵀ
    H_{δξ } =  Rᵀ [ᴳv]_×

This filter is implemented in numpy because it is invoked once per IMU
sample at evaluation time and does not need to be differentiated.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from utils import GRAVITY, so3_exp_np, so3_hat_np


def _skew(v: np.ndarray) -> np.ndarray:
    return np.array(
        [[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]],
        dtype=np.float64,
    )


@dataclass
class VelocityEKF:
    """Tightly coupled IMU + body-frame-velocity EKF."""

    R: np.ndarray = field(default_factory=lambda: np.eye(3))
    v: np.ndarray = field(default_factory=lambda: np.zeros(3))
    p: np.ndarray = field(default_factory=lambda: np.zeros(3))
    b_a: np.ndarray = field(default_factory=lambda: np.zeros(3))
    b_g: np.ndarray = field(default_factory=lambda: np.zeros(3))
    P: np.ndarray = field(default_factory=lambda: np.eye(15) * 1e-3)
    # Bias random walk noise density (per-second variance). Hand-tuned
    # values that work for typical MEMS IMUs; override at construction.
    sigma_ba_rw: float = 1e-3
    sigma_bg_rw: float = 1e-4
    gravity: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, -GRAVITY])
    )

    # ------------------------------------------------------------------
    def reset(
        self,
        R: np.ndarray | None = None,
        v: np.ndarray | None = None,
        p: np.ndarray | None = None,
        P: np.ndarray | None = None,
    ) -> None:
        """Reset the filter to a known initial state."""
        if R is not None:
            self.R = R.astype(np.float64).copy()
        else:
            self.R = np.eye(3)
        self.v = (v if v is not None else np.zeros(3)).astype(np.float64).copy()
        self.p = (p if p is not None else np.zeros(3)).astype(np.float64).copy()
        self.b_a = np.zeros(3)
        self.b_g = np.zeros(3)
        self.P = (P if P is not None else np.eye(15) * 1e-3).astype(np.float64).copy()

    # ------------------------------------------------------------------
    def predict(
        self,
        acc: np.ndarray,
        gyro: np.ndarray,
        dt: float,
        imu_log_var: np.ndarray | None = None,
    ) -> None:
        """Propagate state and covariance with one corrected IMU sample.

        Args:
            acc: ``(3,)`` corrected body-frame accel (output of AirIMU).
            gyro: ``(3,)`` corrected body-frame gyro.
            dt: timestep in seconds.
            imu_log_var: optional ``(6,)`` log-variance from AirIMU
                (first three accel, last three gyro). If ``None`` a
                conservative default is used.
        """
        # Bias-corrected measurements.
        omega = gyro - self.b_g
        acc_b = acc - self.b_a

        # Nominal state propagation (Eq. 7).
        R_prev = self.R
        v_prev = self.v
        a_world = R_prev @ acc_b + self.gravity

        self.R = R_prev @ so3_exp_np(omega * dt)
        self.v = v_prev + a_world * dt
        self.p = self.p + v_prev * dt + 0.5 * a_world * dt * dt
        # Biases stay constant in the nominal model; their random walk
        # appears only in the covariance.

        # ----- linearised error-state dynamics -----
        F = np.eye(15)
        # δξ <- δξ - R * δb_g * dt   (rotation error grows from gyro bias)
        F[0:3, 12:15] = -R_prev * dt
        # δv <- δv - R [a_b]_× δξ dt - R δb_a dt
        F[3:6, 0:3] = -R_prev @ _skew(acc_b) * dt
        F[3:6, 9:12] = -R_prev * dt
        # δp <- δp + δv dt
        F[6:9, 3:6] = np.eye(3) * dt

        # Process noise.
        if imu_log_var is None:
            sigma_a = 0.05
            sigma_g = 0.005
            var_a = np.full(3, sigma_a ** 2)
            var_g = np.full(3, sigma_g ** 2)
        else:
            # log_var stores log of σ². AirIMU outputs both accel (first 3)
            # and gyro (last 3) blocks.
            var_a = np.exp(np.asarray(imu_log_var[:3], dtype=np.float64))
            var_g = np.exp(np.asarray(imu_log_var[3:], dtype=np.float64))

        Q = np.zeros((15, 15))
        # Accel noise enters δv via R: cov += R diag(var_a) Rᵀ * dt²
        Q[3:6, 3:6] = R_prev @ np.diag(var_a) @ R_prev.T * dt * dt
        # Gyro noise enters δξ: cov += diag(var_g) * dt²
        Q[0:3, 0:3] = np.diag(var_g) * dt * dt
        # Bias random walks.
        Q[9:12, 9:12] = np.eye(3) * (self.sigma_ba_rw ** 2) * dt
        Q[12:15, 12:15] = np.eye(3) * (self.sigma_bg_rw ** 2) * dt

        self.P = F @ self.P @ F.T + Q
        # Symmetrise to keep numerical noise from breaking PSD.
        self.P = 0.5 * (self.P + self.P.T)

    # ------------------------------------------------------------------
    def update_velocity(
        self,
        v_body_meas: np.ndarray,
        log_var: np.ndarray,
    ) -> None:
        """Body-frame velocity measurement update.

        Args:
            v_body_meas: ``(3,)`` AirIO-predicted body-frame velocity.
            log_var: ``(3,)`` AirIO-predicted diagonal log-variance.
        """
        v_body_pred = self.R.T @ self.v
        innovation = v_body_meas - v_body_pred  # (3,)

        # Jacobians (Eq. 11).
        H = np.zeros((3, 15))
        H[:, 0:3] = self.R.T @ _skew(self.v)  # ∂h/∂δξ
        H[:, 3:6] = self.R.T                  # ∂h/∂δᴳv

        R_meas = np.diag(np.exp(np.asarray(log_var, dtype=np.float64)))
        S = H @ self.P @ H.T + R_meas
        K = self.P @ H.T @ np.linalg.inv(S)
        delta = K @ innovation  # (15,)

        # Apply error state to nominal state.
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

        # Joseph form for covariance update (numerically nicer than (I-KH)P).
        I_KH = np.eye(15) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R_meas @ K.T
        self.P = 0.5 * (self.P + self.P.T)

    # ------------------------------------------------------------------
    def get_pose(self) -> np.ndarray:
        """Return the current pose as a 4x4 homogeneous transform."""
        T = np.eye(4)
        T[:3, :3] = self.R
        T[:3, 3] = self.p
        return T
