"""
Cross-attention tight fusion of BranchA and AirIO features.

Each modality contributes one token per timestep (after the same
projection to a shared feature dimension. The two streams are concatenated
along the time axis with a learned modality embedding so the
transformer can tell them apart, and a stack of standard transformer
encoder layers performs *self*-attention over the combined ``2T``
tokens — which is mathematically equivalent to alternating
self-attention within each modality and cross-attention between them.

The final per-timestep fused representation is read out by averaging
the two tokens at each timestep (``vision_t`` and ``imu_t``) and
running it through the same pose head used in the other fusion
branches.

The model is heavier than the gated variant (a few million extra
parameters) but captures fine-grained per-token cross-modal
interactions — at the cost of more data to train and harder
interpretation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from imu_only.model import AirIONet  # noqa: E402
from vision_only.model import BranchA  # noqa: E402
from vision_only.utils import gram_schmidt  # noqa: E402


class _ImuFeatureExtractor(nn.Module):
    """Reuse AirIO's CNN + bi-GRU as a per-sample feature extractor."""

    def __init__(self, airio: AirIONet) -> None:
        super().__init__()
        self.airio = airio
        self.feat_dim = 2 * airio.gru.hidden_size

    def forward(
        self, acc: torch.Tensor, gyro: torch.Tensor, attitude: torch.Tensor
    ) -> torch.Tensor:
        imu = torch.cat([acc, gyro], dim=-1)
        f_imu = self.airio.imu_cnn(imu)
        f_att = self.airio.att_cnn(attitude)
        feat = torch.cat([f_imu, f_att], dim=-1)
        seq, _ = self.airio.gru(feat)
        return seq


class CrossAttentionFusionNet(nn.Module):
    """Tight fusion of vision + IMU via a cross-modal transformer.

    Args:
        branch_a / airio: optional pretrained backbones.
        feat_dim: shared transformer model dimension.
        num_heads: multi-head attention heads.
        num_layers: number of stacked transformer encoder blocks.
        ffn_hidden: feed-forward hidden width inside each block.
        dropout: dropout used inside the transformer.
        max_len: maximum sequence length supported by positional emb.
        freeze_backbones: freeze ``BranchA`` + ``AirIONet`` for a warm-up.
    """

    def __init__(
        self,
        branch_a: BranchA | None = None,
        airio: AirIONet | None = None,
        feat_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        ffn_hidden: int = 256,
        dropout: float = 0.1,
        max_len: int = 256,
        freeze_backbones: bool = False,
    ) -> None:
        super().__init__()
        self.branch_a = branch_a if branch_a is not None else BranchA()
        self.airio = airio if airio is not None else AirIONet()
        self._imu_features = _ImuFeatureExtractor(self.airio)

        self.vis_proj = nn.Linear(128, feat_dim)
        self.imu_proj = nn.Linear(self._imu_features.feat_dim, feat_dim)

        # Learned modality + temporal embeddings.
        self.modality_emb = nn.Parameter(torch.zeros(2, feat_dim))
        self.pos_emb = nn.Parameter(torch.zeros(max_len, feat_dim))
        nn.init.trunc_normal_(self.modality_emb, std=0.02)
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feat_dim,
            nhead=num_heads,
            dim_feedforward=ffn_hidden,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fused_norm = nn.LayerNorm(feat_dim)

        head_hidden = 256
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
        hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Predict per-pair relative pose with cross-modal attention.

        Args:
            hidden: optional BranchA LSTM state ``(h_n, c_n)`` from the
                previous chunk. ``None`` resets the LSTM (start of sequence
                or random training batch).

        Returns a dict with keys matching ``GatedFusionNet`` plus
        ``vis_imu_cos`` (cosine similarity diagnostic) and ``hidden``.
        """
        B, T = frames_t.shape[:2]
        W = imu_acc.shape[2]

        _, _, _, hidden_out, vis_feat = self.branch_a(frames_t, frames_t1, hidden=hidden)

        imu_acc_flat = imu_acc.reshape(B * T, W, 3)
        imu_gyro_flat = imu_gyro.reshape(B * T, W, 3)
        att_flat = attitude.reshape(B * T, W, 3)
        imu_seq = self._imu_features(imu_acc_flat, imu_gyro_flat, att_flat)
        imu_last = imu_seq[:, -1, :].view(B, T, -1)

        v = self.vis_proj(vis_feat)               # [B, T, D]
        u = self.imu_proj(imu_last)               # [B, T, D]

        # Add modality and temporal embeddings.
        if T > self.pos_emb.shape[0]:
            raise RuntimeError(
                f"Sequence length {T} exceeds max_len={self.pos_emb.shape[0]}; "
                f"increase the constructor's max_len."
            )
        pos = self.pos_emb[:T].unsqueeze(0)       # [1, T, D]
        v = v + pos + self.modality_emb[0]
        u = u + pos + self.modality_emb[1]

        # Concatenate along the time axis: [B, 2T, D].
        tokens = torch.cat([v, u], dim=1)
        tokens = self.transformer(tokens)
        tokens = self.fused_norm(tokens)

        v_out = tokens[:, :T, :]
        u_out = tokens[:, T:, :]
        fused = 0.5 * (v_out + u_out)

        trans = self.trans_head(fused)
        rot_6d = self.rot_head(fused)
        R = gram_schmidt(rot_6d)

        # Diagnostic: cosine similarity per timestep between vision and
        # IMU output tokens — proxy for "are the modalities agreeing?"
        cos = torch.nn.functional.cosine_similarity(v_out, u_out, dim=-1)

        return {
            "trans": trans,
            "rot_6d": rot_6d,
            "R": R,
            "fused": fused,
            "vis_token": v_out,
            "imu_token": u_out,
            "vis_imu_cos": cos,
            "hidden": hidden_out,
        }

    # ------------------------------------------------------------------
    @staticmethod
    def load_pretrained(
        vision_checkpoint: str | None = None,
        airio_checkpoint: str | None = None,
        device: torch.device | str = "cpu",
        freeze_backbones: bool = False,
        **kwargs,
    ) -> "CrossAttentionFusionNet":
        branch_a = BranchA().to(device)
        if vision_checkpoint:
            st = torch.load(vision_checkpoint, map_location=device)
            branch_a.load_state_dict(st.get("model", st))
        airio = AirIONet().to(device)
        if airio_checkpoint:
            st = torch.load(airio_checkpoint, map_location=device)
            airio.load_state_dict(st.get("model", st))
        net = CrossAttentionFusionNet(
            branch_a=branch_a, airio=airio,
            freeze_backbones=freeze_backbones,
            **kwargs,
        ).to(device)
        return net

    def num_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
