"""
Gated tight fusion of BranchA (vision) and AirIO (IMU) features.

Per timestep the network produces a per-dimension gate ``g ∈ [0, 1]``
that softly mixes a vision feature vector ``f_v`` and an IMU feature
vector ``f_i`` of the same dimensionality:

    fused = g ⊙ f_v + (1 - g) ⊙ f_i

The gate is computed from the concatenation of both features through a
small MLP. The fused feature is decoded into translation and a 6D
rotation by two small MLP heads — same convention as ``vision_only``.

Sub-network behaviour:
    - ``BranchA`` is reused **as-is** (no architectural changes). Its
      already-exposed ``features [B, T, 128]`` post-FC tensor is the
      vision contribution. We build the network with ``feat_dim_vis = 128``.
    - ``AirIONet`` is reused but we pull the GRU hidden output before the
      ``velocity_head`` / ``uncertainty_head``. The bidirectional GRU has
      ``2 * gru_hidden = 256`` channels per timestep, projected to
      ``feat_dim_imu = 128`` to match vision.

Loading from pretrained checkpoints is supported and *strongly*
recommended — random init from scratch needs ~10× the data.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

# Sibling-branch imports.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from imu_only.model import AirIONet  # noqa: E402
from imu_only.utils import GRAVITY  # noqa: F401, E402  (re-export for convenience)
from vision_only.model import BranchA  # noqa: E402
from vision_only.utils import gram_schmidt  # noqa: E402


class _ImuFeatureExtractor(nn.Module):
    """Adapter that extracts AirIO's per-sample bi-GRU output as features.

    AirIO's forward returns ``(velocity, log_var)`` only. For tight
    fusion we want the GRU hidden state. Rather than monkey-patching
    AirIO we run its CNN encoders + GRU manually here, sharing the same
    weights via a reference.
    """

    def __init__(self, airio: AirIONet) -> None:
        super().__init__()
        self.airio = airio
        self.feat_dim = 2 * airio.gru.hidden_size  # bi-directional

    def forward(
        self, acc: torch.Tensor, gyro: torch.Tensor, attitude: torch.Tensor
    ) -> torch.Tensor:
        """Return per-sample features ``[B, W, 2 * gru_hidden]``."""
        imu = torch.cat([acc, gyro], dim=-1)
        f_imu = self.airio.imu_cnn(imu)
        f_att = self.airio.att_cnn(attitude)
        feat = torch.cat([f_imu, f_att], dim=-1)
        seq, _ = self.airio.gru(feat)
        return seq


class GatedFusionNet(nn.Module):
    """Tight fusion via per-dimension gating of vision and IMU features.

    Args:
        branch_a: a (possibly pretrained) ``vision_only.model.BranchA``.
        airio: a (possibly pretrained) ``imu_only.model.AirIONet``.
        feat_dim: dimension of the shared fusion space (both vision and
            IMU features are projected here before gating). Defaults to
            128, matching BranchA's output.
        head_hidden: hidden width of the per-dimension gate MLP.
        freeze_backbones: if ``True``, set ``requires_grad=False`` on
            both backbones so only the fusion / heads update. Useful
            for warm-up training before joint fine-tuning.
    """

    def __init__(
        self,
        branch_a: BranchA | None = None,
        airio: AirIONet | None = None,
        feat_dim: int = 128,
        head_hidden: int = 256,
        freeze_backbones: bool = False,
    ) -> None:
        super().__init__()
        self.branch_a = branch_a if branch_a is not None else BranchA()
        self.airio = airio if airio is not None else AirIONet()
        self._imu_features = _ImuFeatureExtractor(self.airio)

        # Project both modalities to ``feat_dim``.
        self.vis_proj = nn.Linear(128, feat_dim)
        self.imu_proj = nn.Linear(self._imu_features.feat_dim, feat_dim)

        # Gate MLP — input is the concatenation of the two projected
        # features, output is a per-dimension gate in [0, 1].
        self.gate = nn.Sequential(
            nn.Linear(2 * feat_dim, head_hidden),
            nn.GELU(),
            nn.Linear(head_hidden, feat_dim),
            nn.Sigmoid(),
        )

        self.fused_norm = nn.LayerNorm(feat_dim)
        self.trans_head = nn.Sequential(
            nn.Linear(feat_dim, head_hidden), nn.GELU(),
            nn.Linear(head_hidden, 3),
        )
        self.rot_head = nn.Sequential(
            nn.Linear(feat_dim, head_hidden), nn.GELU(),
            nn.Linear(head_hidden, 6),
        )
        self._init_heads()

        if freeze_backbones:
            for p in self.branch_a.parameters():
                p.requires_grad_(False)
            for p in self.airio.parameters():
                p.requires_grad_(False)

    def _init_heads(self) -> None:
        for m in (self.vis_proj, self.imu_proj):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
        for head in (self.trans_head, self.rot_head):
            for m in head.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight, gain=0.01)
                    nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    def forward(
        self,
        frames_t: torch.Tensor,
        frames_t1: torch.Tensor,
        imu_acc: torch.Tensor,
        imu_gyro: torch.Tensor,
        attitude: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Predict per-pair relative pose with gated vision + IMU fusion.

        Args:
            frames_t, frames_t1: ``[B, T, 3, H, W]`` consecutive frames.
            imu_acc, imu_gyro: ``[B, T, W, 3]`` IMU context per pair.
            attitude: ``[B, T, W, 3]`` per-sample so(3) attitude.

        Returns:
            A dict with keys:
                ``trans``    ``[B, T, 3]``,
                ``rot_6d``   ``[B, T, 6]``,
                ``R``        ``[B, T, 3, 3]``,
                ``gate``     ``[B, T, feat_dim]`` (for diagnostics),
                ``vis_feat`` ``[B, T, 128]``  (BranchA features),
                ``imu_feat`` ``[B, T, feat_dim]`` (projected IMU features).
        """
        B, T = frames_t.shape[:2]
        W = imu_acc.shape[2]

        # Vision features — BranchA emits ``features [B, T, 128]``.
        _, _, _, _, vis_feat = self.branch_a(frames_t, frames_t1, hidden=None)

        # IMU features — fold (B, T) into the batch, run the bi-GRU,
        # take the last sample of every window as the per-pair feature.
        imu_acc_flat = imu_acc.reshape(B * T, W, 3)
        imu_gyro_flat = imu_gyro.reshape(B * T, W, 3)
        att_flat = attitude.reshape(B * T, W, 3)
        imu_seq = self._imu_features(imu_acc_flat, imu_gyro_flat, att_flat)
        imu_last = imu_seq[:, -1, :].view(B, T, -1)

        v = self.vis_proj(vis_feat)
        u = self.imu_proj(imu_last)
        gate = self.gate(torch.cat([v, u], dim=-1))
        fused = self.fused_norm(gate * v + (1.0 - gate) * u)

        trans = self.trans_head(fused)
        rot_6d = self.rot_head(fused)
        R = gram_schmidt(rot_6d)

        return {
            "trans": trans,
            "rot_6d": rot_6d,
            "R": R,
            "gate": gate,
            "vis_feat": vis_feat,
            "imu_feat": u,
        }

    # ------------------------------------------------------------------
    @staticmethod
    def load_pretrained(
        vision_checkpoint: str | None = None,
        airio_checkpoint: str | None = None,
        device: torch.device | str = "cpu",
        freeze_backbones: bool = False,
    ) -> "GatedFusionNet":
        """Construct a fusion model with optionally pretrained backbones."""
        branch_a = BranchA().to(device)
        if vision_checkpoint:
            st = torch.load(vision_checkpoint, map_location=device)
            branch_a.load_state_dict(st.get("model", st))
        airio = AirIONet().to(device)
        if airio_checkpoint:
            st = torch.load(airio_checkpoint, map_location=device)
            airio.load_state_dict(st.get("model", st))
        net = GatedFusionNet(
            branch_a=branch_a, airio=airio, freeze_backbones=freeze_backbones
        ).to(device)
        return net

    def num_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
