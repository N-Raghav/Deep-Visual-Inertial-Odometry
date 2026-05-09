import torch
import torch.nn as nn

from prgflow.squeezenet import SHead, THead, model_size_mb
from prgflow.warp import compose, decompose_to_pseudosim, pseudo_similarity_matrix, warp_image


class PRGFlow(nn.Module):
    def __init__(self, patch_size=128, base_channels=32, fire_blocks=4, dropout=0.7):
        super().__init__()
        self.patch_size = patch_size
        self.t_head_1 = THead(base_channels=base_channels, fire_blocks=fire_blocks, dropout=dropout)
        self.t_head_2 = THead(base_channels=base_channels, fire_blocks=fire_blocks, dropout=dropout)
        self.s_head_1 = SHead(base_channels=base_channels, fire_blocks=fire_blocks, dropout=dropout)
        self.s_head_2 = SHead(base_channels=base_channels, fire_blocks=fire_blocks, dropout=dropout)

    def _delta_to_matrix(self, delta, kind, H, W):
        B = delta.shape[0]
        zeros = torch.zeros(B, device=delta.device, dtype=delta.dtype)
        if kind == "t":
            return pseudo_similarity_matrix(zeros, delta[:, 0], delta[:, 1], H, W)
        return pseudo_similarity_matrix(delta[:, 0], zeros, zeros, H, W)

    def forward(self, patch_t, patch_t1, return_matrix=False):
        if patch_t.ndim == 3:
            patch_t = patch_t.unsqueeze(0)
        if patch_t1.ndim == 3:
            patch_t1 = patch_t1.unsqueeze(0)
        if patch_t.shape != patch_t1.shape:
            raise ValueError(f"shape mismatch: {patch_t.shape} vs {patch_t1.shape}")

        B, _, H, W = patch_t.shape
        H_total = torch.eye(3, device=patch_t.device, dtype=patch_t.dtype).unsqueeze(0).repeat(B, 1, 1)

        heads = [
            (self.t_head_1, "t"),
            (self.s_head_1, "s"),
            (self.t_head_2, "t"),
            (self.s_head_2, "s"),
        ]

        for head, kind in heads:
            warped = warp_image(patch_t1, H_total)
            stacked = torch.cat([patch_t, warped], dim=1)
            delta = head(stacked)
            H_delta = self._delta_to_matrix(delta, kind, H, W)
            H_total = compose(H_total, H_delta)

        s, tx, ty = decompose_to_pseudosim(H_total, H, W)
        pred = torch.stack([s, tx, ty], dim=1)
        if return_matrix:
            return pred, H_total
        return pred

    def num_trainable_parameters(self):
        total = 0
        for p in self.parameters():
            if p.requires_grad:
                total += p.numel()
        return total

    def size_mb(self):
        return model_size_mb(self)
