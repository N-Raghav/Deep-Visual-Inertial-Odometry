"""
Networks used by the IMU-only branch.

Two architectures are provided, both built from the same building blocks
(``CNNEncoder1D`` + bidirectional GRU + MLP heads):

    - ``AirIMUNet`` predicts a per-sample IMU correction and uncertainty
      (gyro + accel) from the raw IMU stream. It plays the role of the
      "AirIMU" component described in Qiu et al. 2023 (the pre-integration
      pre-processing stage that feeds into AirIO's EKF).

    - ``AirIONet`` is the motion network of Qiu et al. 2025 ("AirIO:
      Learning Inertial Odometry with Enhanced IMU Feature Observability",
      RA-L 2025). It takes corrected body-frame IMU together with the
      attitude in ``so(3)`` and outputs the body-frame velocity and its
      diagonal uncertainty.

Both networks operate on windows of length ``W`` and produce per-sample
outputs (``[B, W, ...]``). The CNN1D layers preserve the time dimension
(``padding="same"``) so the GRU sees one feature vector per IMU sample.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CNNEncoder1D(nn.Module):
    """Stack of 1D conv blocks (Conv1d -> BN1d -> GELU) ending in dropout.

    Input shape ``[B, T, C_in]`` is internally transposed to ``[B, C_in, T]``
    for ``nn.Conv1d`` and transposed back before returning.

    Args:
        in_channels: Number of input channels (6 for IMU, 3 for attitude).
        channels: Channel counts of successive conv layers, e.g.
            ``[64, 128, 128]`` produces three blocks 6→64→128→128.
        kernel_size: Convolution kernel length, defaults to 3.
        dropout: Dropout probability applied at the end of the stack.
    """

    def __init__(
        self,
        in_channels: int,
        channels: list[int],
        kernel_size: int = 3,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_channels
        pad = kernel_size // 2
        for c in channels:
            layers += [
                nn.Conv1d(prev, c, kernel_size=kernel_size, padding=pad, bias=False),
                nn.BatchNorm1d(c),
                nn.GELU(),
            ]
            prev = c
        layers.append(nn.Dropout(p=dropout))
        self.net = nn.Sequential(*layers)
        self.out_channels = prev

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a length-``T`` sequence of ``C_in``-dim feature vectors.

        Args:
            x: ``[B, T, C_in]``.
        Returns:
            ``[B, T, C_out]`` features.
        """
        x = x.transpose(1, 2)        # [B, C_in, T]
        x = self.net(x)              # [B, C_out, T]
        return x.transpose(1, 2)     # [B, T, C_out]


class _MLPHead(nn.Module):
    """Per-timestep MLP applied to the GRU output."""

    def __init__(self, in_dim: int, out_dim: int, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# AirIMU
# ---------------------------------------------------------------------------
class AirIMUNet(nn.Module):
    """IMU-correction network (Qiu et al. 2023, ref [30] of the AirIO paper).

    For every IMU sample it predicts:
        - ``correction`` ``[B, W, 6]``: additive correction terms for accel
          and gyro. The corrected IMU is ``â = a + σ̂_a``,
          ``ŵ = ω + σ̂_g``.
        - ``log_var``   ``[B, W, 6]``: log of the diagonal IMU noise
          variance (3 for accel, 3 for gyro). These are fed to the EKF as
          the per-frame measurement covariance.

    The architecture is a CNN encoder followed by a bidirectional GRU and
    two MLP heads. No attitude information is supplied — AirIMU operates
    purely on the raw IMU stream.
    """

    def __init__(
        self,
        cnn_channels: tuple[int, ...] = (64, 128, 128),
        gru_hidden: int = 128,
        gru_layers: int = 2,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.cnn = CNNEncoder1D(in_channels=6, channels=list(cnn_channels), dropout=dropout)
        self.gru = nn.GRU(
            input_size=self.cnn.out_channels,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if gru_layers > 1 else 0.0,
        )
        feat_dim = 2 * gru_hidden
        self.correction_head = _MLPHead(feat_dim, 6)
        self.uncertainty_head = _MLPHead(feat_dim, 6)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Small init on heads so early-training corrections don't blow up.
        for head in (self.correction_head, self.uncertainty_head):
            for m in head.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight, gain=0.01)
                    nn.init.zeros_(m.bias)

    def forward(
        self, acc: torch.Tensor, gyro: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict per-sample IMU corrections and uncertainty.

        Args:
            acc: ``[B, W, 3]`` raw accelerometer samples.
            gyro: ``[B, W, 3]`` raw gyroscope samples.
        Returns:
            Tuple ``(correction, log_var)``, each ``[B, W, 6]`` with the
            first three columns referring to accel and the last three to
            gyro.
        """
        x = torch.cat([acc, gyro], dim=-1)
        feat = self.cnn(x)
        seq, _ = self.gru(feat)
        correction = self.correction_head(seq)
        log_var = self.uncertainty_head(seq)
        # Clamp log-variance to a sensible range to prevent NaNs in NLL.
        log_var = log_var.clamp(min=-10.0, max=10.0)
        return correction, log_var

    def correct(
        self, acc: torch.Tensor, gyro: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convenience wrapper that returns the corrected IMU plus uncertainty.

        Returns:
            ``(acc_hat, gyro_hat, log_var)`` where ``acc_hat = acc + σ̂_a``
            and ``gyro_hat = gyro + σ̂_g``.
        """
        correction, log_var = self.forward(acc, gyro)
        acc_hat = acc + correction[..., :3]
        gyro_hat = gyro + correction[..., 3:]
        return acc_hat, gyro_hat, log_var

    def num_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# AirIO
# ---------------------------------------------------------------------------
class AirIONet(nn.Module):
    """AirIO motion network (Qiu et al. 2025).

    Takes corrected body-frame IMU together with the drone's attitude and
    predicts the body-frame velocity ``ᴮv`` plus its diagonal Gaussian
    uncertainty per IMU sample.

    Two encoders run in parallel:
        - an IMU encoder (3+3 → channels) on stacked ``[acc | gyro]``
        - an attitude encoder (3 → channels/2) on the ``so(3)`` log of the
          orientation.
    Their outputs are concatenated along the channel dimension and fed
    into a bidirectional GRU. Two MLP heads then produce velocity and
    log-variance per timestep.
    """

    def __init__(
        self,
        imu_cnn_channels: tuple[int, ...] = (64, 128, 128),
        att_cnn_channels: tuple[int, ...] = (32, 64),
        gru_hidden: int = 128,
        gru_layers: int = 2,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.imu_cnn = CNNEncoder1D(
            in_channels=6, channels=list(imu_cnn_channels), dropout=dropout
        )
        self.att_cnn = CNNEncoder1D(
            in_channels=3, channels=list(att_cnn_channels), dropout=dropout
        )
        feat_in = self.imu_cnn.out_channels + self.att_cnn.out_channels
        self.gru = nn.GRU(
            input_size=feat_in,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if gru_layers > 1 else 0.0,
        )
        feat_dim = 2 * gru_hidden
        self.velocity_head = _MLPHead(feat_dim, 3)
        self.uncertainty_head = _MLPHead(feat_dim, 3)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        for head in (self.velocity_head, self.uncertainty_head):
            for m in head.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight, gain=0.01)
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        acc: torch.Tensor,
        gyro: torch.Tensor,
        attitude: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict body-frame velocity and its diagonal log-variance.

        Args:
            acc: ``[B, W, 3]`` body-frame accel (already corrected by
                AirIMU during training and inference).
            gyro: ``[B, W, 3]`` body-frame gyro.
            attitude: ``[B, W, 3]`` per-sample attitude ``ξ = log_SO(3)(R)``.

        Returns:
            Tuple ``(v_body, log_var)`` of shape ``[B, W, 3]``.
        """
        imu = torch.cat([acc, gyro], dim=-1)
        f_imu = self.imu_cnn(imu)        # [B, W, C_imu]
        f_att = self.att_cnn(attitude)   # [B, W, C_att]
        feat = torch.cat([f_imu, f_att], dim=-1)
        seq, _ = self.gru(feat)
        v_body = self.velocity_head(seq)
        log_var = self.uncertainty_head(seq).clamp(min=-10.0, max=10.0)
        return v_body, log_var

    def num_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
