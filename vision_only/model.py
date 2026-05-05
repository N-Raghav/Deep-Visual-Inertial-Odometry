"""
Branch A — Vision-Only Odometry network.

Implements the four-stage architecture described in the README:

    Stage 1:  Siamese MobileNetV2 encoder  (shared weights)
    Stage 2:  Correlation layer            (max_displacement=4)
    Stage 3:  Spatial compression          (concat + conv + pool)
    Stage 4:  LSTM + separate translation / rotation heads

Public classes:
    - CNNEncoder           Per-frame feature extractor.
    - CorrelationLayer     Vectorized cross-frame correlation.
    - BranchA              Full network used by ``train.py`` / ``evaluate.py``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v2

try:
    from vision_only.utils import gram_schmidt
except ImportError:
    from utils import gram_schmidt


class CNNEncoder(nn.Module):
    """Siamese feature extractor based on MobileNetV2.

    Uses the first 14 layers of MobileNetV2's ``features`` block (indices
    ``0..13`` inclusive), giving a spatial stride of 16 and 96 output
    channels. A 1x1 convolution followed by BatchNorm and ReLU reduces
    the channel count to 64 to keep the correlation layer cheap.
    """

    def __init__(self, out_channels: int = 64) -> None:
        super().__init__()
        backbone = mobilenet_v2(weights=None)
        # Layers 0..13 -> spatial stride 16, 96 channels.
        self.features = nn.Sequential(*list(backbone.features[:14]))
        self.reduce = nn.Sequential(
            nn.Conv2d(96, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features for a single frame.

        Args:
            x: Image tensor of shape ``[B, 3, H, W]`` (ImageNet-normalized).

        Returns:
            Feature map of shape ``[B, out_channels, H/16, W/16]``.
        """
        feat = self.features(x)
        feat = self.reduce(feat)
        return feat


class CorrelationLayer(nn.Module):
    """Vectorized cross-frame correlation with bounded displacement.

    For each location ``(i, j)`` in ``feat_t`` the layer computes the
    mean-over-channels dot product against every location in ``feat_t1``
    inside a ``(2 * max_displacement + 1)`` square neighborhood. The
    output therefore has ``(2 * D + 1) ** 2`` channels, one per offset.

    The implementation uses ``F.unfold`` to materialise the neighborhood
    as a single tensor and then reduces with a batched matrix multiply,
    avoiding any Python-level loops over spatial offsets.
    """

    def __init__(self, max_displacement: int = 4) -> None:
        super().__init__()
        self.max_displacement = max_displacement
        self.kernel = 2 * max_displacement + 1
        self.num_offsets = self.kernel ** 2

    def forward(self, feat_t: torch.Tensor, feat_t1: torch.Tensor) -> torch.Tensor:
        """Compute the correlation volume.

        Args:
            feat_t: Features at time ``t``, shape ``[B, C, H, W]``.
            feat_t1: Features at time ``t+1``, shape ``[B, C, H, W]``.

        Returns:
            Tensor of shape ``[B, K*K, H, W]`` where ``K = 2*D + 1``.
        """
        b, c, h, w = feat_t.shape
        d = self.max_displacement
        k = self.kernel

        # Pad feat_t1 so that every offset within +/- d is well-defined.
        feat_t1_pad = F.pad(feat_t1, (d, d, d, d))
        # Unfold all displaced patches: [B, C * K * K, H * W].
        patches = F.unfold(feat_t1_pad, kernel_size=k, padding=0, stride=1)
        # Reshape to [B, C, K*K, H*W].
        patches = patches.view(b, c, k * k, h * w)

        # feat_t flattened to [B, C, H*W].
        feat_t_flat = feat_t.view(b, c, h * w)

        # Mean-over-channels dot product per displacement.
        # [B, C, 1, H*W] * [B, C, K*K, H*W] -> [B, K*K, H*W].
        corr = (feat_t_flat.unsqueeze(2) * patches).mean(dim=1)
        corr = corr.view(b, k * k, h, w)
        return corr


class BranchA(nn.Module):
    """Vision-only odometry network described in ``README.md``.

    Forward pass takes a sequence of frame pairs ``[B, T, 3, H, W]`` and
    returns per-step translation, 6D rotation, full 3x3 rotation matrix,
    the updated LSTM hidden state, and a 128-dim feature vector for use
    by downstream Branch C visual-inertial fusion.
    """

    def __init__(
        self,
        feat_channels: int = 64,
        max_displacement: int = 4,
        lstm_hidden: int = 256,
        lstm_layers: int = 2,
        lstm_dropout: float = 0.3,
        fc_dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.feat_channels = feat_channels
        self.max_displacement = max_displacement
        self.lstm_hidden = lstm_hidden
        self.lstm_layers = lstm_layers

        # Stage 1: Siamese CNN.
        self.encoder = CNNEncoder(out_channels=feat_channels)

        # Stage 2: Correlation + compression.
        self.correlation = CorrelationLayer(max_displacement=max_displacement)
        num_offsets = (2 * max_displacement + 1) ** 2
        self.corr_compress = nn.Sequential(
            nn.Conv2d(num_offsets, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, feat_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(feat_channels),
            nn.ReLU(inplace=True),
        )

        # Stage 3: Spatial compression of the concatenated tensor.
        concat_channels = feat_channels * 3  # corr + feat_t + feat_t1
        self.spatial_compress = nn.Sequential(
            nn.Conv2d(concat_channels, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, feat_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feat_channels),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.flat_dim = feat_channels * 4 * 4  # 1024

        # Stage 4: LSTM and pose heads.
        self.lstm = nn.LSTM(
            input_size=self.flat_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=lstm_dropout if lstm_layers > 1 else 0.0,
            bidirectional=False,
        )
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=fc_dropout),
        )
        self.trans_head = nn.Linear(128, 3)
        self.rot_head = nn.Linear(128, 6)

        self._init_weights()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------
    def _init_weights(self) -> None:
        """Initialize convolutional, LSTM and pose-head weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # LSTM-specific init.
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

        # Small init on pose heads for stable early training.
        nn.init.xavier_uniform_(self.trans_head.weight, gain=0.01)
        nn.init.zeros_(self.trans_head.bias)
        nn.init.xavier_uniform_(self.rot_head.weight, gain=0.01)
        nn.init.zeros_(self.rot_head.bias)

    # ------------------------------------------------------------------
    # Helper: encode one frame pair into a 1024-dim feature.
    # ------------------------------------------------------------------
    def encode_frame_pair(
        self, frame_t: torch.Tensor, frame_t1: torch.Tensor
    ) -> torch.Tensor:
        """Encode a single pair of frames (no sequence dimension).

        Args:
            frame_t: Tensor of shape ``[B, 3, H, W]``.
            frame_t1: Tensor of shape ``[B, 3, H, W]``.

        Returns:
            Flat per-pair feature of shape ``[B, 1024]``.
        """
        feat_t = self.encoder(frame_t)
        feat_t1 = self.encoder(frame_t1)
        corr = self.correlation(feat_t, feat_t1)
        corr = self.corr_compress(corr)
        x = torch.cat([corr, feat_t, feat_t1], dim=1)
        x = self.spatial_compress(x)
        x = self.pool(x)
        x = torch.flatten(x, start_dim=1)
        return x

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        frames_t: torch.Tensor,
        frames_t1: torch.Tensor,
        hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        tuple[torch.Tensor, torch.Tensor],
        torch.Tensor,
    ]:
        """Run the full network over a sequence of frame pairs.

        Args:
            frames_t: ``[B, T, 3, H, W]`` frames at time ``t``.
            frames_t1: ``[B, T, 3, H, W]`` frames at time ``t+1``.
            hidden: Optional ``(h_n, c_n)`` LSTM state. ``None`` resets the
                state, which is the correct behaviour at the start of a new
                trajectory.

        Returns:
            Tuple ``(trans, rot_6d, R, hidden, features)`` where:
                - ``trans``    ``[B, T, 3]`` translations in metric units,
                - ``rot_6d``   ``[B, T, 6]`` raw 6D rotation outputs,
                - ``R``        ``[B, T, 3, 3]`` orthonormal rotations,
                - ``hidden``   updated LSTM state ``(h_n, c_n)``,
                - ``features`` ``[B, T, 128]`` post-FC features for fusion.
        """
        b, t = frames_t.shape[:2]

        # Fold the sequence dimension into the batch for CNN/correlation.
        ft = frames_t.reshape(b * t, *frames_t.shape[2:])
        ft1 = frames_t1.reshape(b * t, *frames_t1.shape[2:])

        flat = self.encode_frame_pair(ft, ft1)         # [B*T, 1024]
        flat = flat.view(b, t, self.flat_dim)          # [B, T, 1024]

        lstm_out, hidden = self.lstm(flat, hidden)     # [B, T, hidden]
        features = self.fc(lstm_out)                   # [B, T, 128]

        trans = self.trans_head(features)              # [B, T, 3]
        rot_6d = self.rot_head(features)               # [B, T, 6]
        R = gram_schmidt(rot_6d)                       # [B, T, 3, 3]

        return trans, rot_6d, R, hidden, features

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------
    def num_trainable_parameters(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
