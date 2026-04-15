"""
PlaneCycleOp – core operator for PlaneCycle.

Applies a pretrained 2D block to one plane of a 3D volume.
Planes: HW (axial, slices along D) / DW (coronal, along H) / DH (sagittal, along W).
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from planecycle.operators.utils import adaptive_avg_pool_along_dim

# For each plane, store (slice_axis, rope_row_axis, rope_col_axis) in (B, D, H, W, C).
# The plane name encodes its two active dims; the remaining dim is the slice axis.
PLANE_TO_AXES: dict[str, tuple[int, int, int]] = {
    "HW": (1, 2, 3),  # slice along D(1), RoPE over H(2) x W(3)
    "DW": (2, 1, 3),  # slice along H(2), RoPE over D(1) x W(3)
    "DH": (3, 1, 2),  # slice along W(3), RoPE over D(1) x H(2)
}


class PlaneCycleOp(nn.Module):
    """Apply a pretrained 2D block to one plane of a 3D volume.

    Three steps: Reshape -> Apply -> Reshape.

    Args:
        block: The wrapped 2D block (transformer block or conv layer).
        blk_idx: Global block index, determines the cycling plane.
        n_blocks: Total number of blocks in the backbone.
        backbone_name: One of 'dinov3' or 'convnext', selects the forward path.
        cycle_order: Ordered plane labels cycled round-robin across blocks.
        rope_embed: RoPE module from the ViT backbone (dinov3 only).
        pool_method: Global token pooling, 'PCg' adaptive avg (recommended) or 'PCm' mean (dinov3 only).
    """

    def __init__(
            self,
            block,
            blk_idx: int,
            n_blocks: int,
            backbone_name: str,
            cycle_order: Tuple[str, ...] = ("HW", "DH", "DW"),
            rope_embed: Optional[nn.Module] = None,
            pool_method="",
    ) -> None:
        super().__init__()
        if backbone_name == 'vit' and pool_method not in ("PCg", "PCm"):
            raise ValueError(f"pool_method must be 'PCg' or 'PCm', got {pool_method!r}")
        self.block = block
        self.blk_idx = blk_idx
        self.n_blocks = n_blocks
        self.backbone_name = backbone_name
        self.rope_embed = rope_embed
        self.pool_method = pool_method
        plane = cycle_order[blk_idx % len(cycle_order)]
        self.plane_dim, self.rope_row, self.rope_col = PLANE_TO_AXES[plane]

    def forward(self, xf: Tensor, xg: Optional[Tensor] = None):
        if self.backbone_name == 'vit':
            return self._forward_vit(xf, xg)
        elif self.backbone_name == 'convnext':
            return self._forward_convnext(xf)
        else:
            raise ValueError(f"Unknown backbone_name '{self.backbone_name}'.")

    def _forward_vit(self, xf: Tensor, xg: Tensor):
        """ViT path: operates on sequence tokens alongside global (CLS) tokens.

        Args:
            xf: Spatial tokens (B, D, H, W, C).
            xg: Global tokens (B, P', g_len, C).
        Returns:
            xf: (B, D, H, W, C), xg: (B, D, g_len, C) on last block else (B, P, g_len, C)
        """
        B, D, *_, C = xf.shape
        P, g_len = xf.shape[self.plane_dim], xg.size(2)

        xf_plane = xf.movedim(self.plane_dim, 1)  # (B, P, S1, S2, C)
        x_seq = xf_plane.reshape(B * P, -1, C)  # (B*P, L, C)

        if self.pool_method == "PCm":
            xg_pooled = xg.mean(dim=1, keepdim=True).expand(-1, P, -1, -1)
        else:
            xg_pooled = adaptive_avg_pool_along_dim(xg, output_size=P, dim=1)
        g_seq = xg_pooled.reshape(B * P, g_len, C)  # (B*P, g_len, C)

        rope = self.rope_embed(H=xf.shape[self.rope_row], W=xf.shape[self.rope_col]) \
            if self.rope_embed else None
        tokens = self.block(torch.cat([g_seq, x_seq], dim=1), rope)  # (B*P, g_len+L, C)

        xf = tokens[:, g_len:].reshape_as(xf_plane).movedim(1, self.plane_dim)
        xg = tokens[:, :g_len].reshape(B, P, g_len, C)
        if self.blk_idx == self.n_blocks - 1:
            xg = adaptive_avg_pool_along_dim(xg, output_size=D, dim=1)  # (B, D, g_len, C)
        return xf, xg

    def _forward_convnext(self, xf: Tensor):
        """CNN path: applies the ConvNeXt block to plane-wise slices.

        Args:
            xf: (B, D, H, W, C)
        Returns:
            xf: (B, D, H, W, C)
        """
        xf_plane = xf.movedim(self.plane_dim, 1)  # (B, P, S1, S2, C)
        xf = self.block(xf_plane.flatten(0, 1).permute(0, 3, 1, 2))  # (B*P, C, S1, S2)
        return xf.permute(0, 2, 3, 1).reshape_as(xf_plane).movedim(1, self.plane_dim)
