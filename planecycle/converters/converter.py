"""
PlaneCycle converter – unified interface for ViT and CNN backbones.

The 2D backbone is kept intact and unmodified.

"""

from typing import Literal, Tuple

import torch.nn as nn
from torch import Tensor

from planecycle.operators.planecycle_op import PLANE_TO_AXES, PlaneCycleOp
from planecycle.operators.utils import adaptive_avg_pool_along_dim

# Supported backbone names and the attributes used to detect each:
#   dinov3 – ViT with RoPE positional encoding and storage tokens
#   convnext – ConvNeXt; iterates backbone.stages
SUPPORTED = ('dinov3', 'convnext')


def _detect_backbone(backbone: nn.Module) -> str:
    """Auto-detect a backbone name from its attributes."""
    if hasattr(backbone, 'blocks') and hasattr(backbone, 'rope_embed'):
        return 'dinov3'
    if hasattr(backbone, 'stages'):
        return 'convnext'
    raise ValueError(
        f"Cannot auto-detect backbone. Supported: {SUPPORTED}."
    )


class PlaneCycleConverter(nn.Module):
    """Wraps a 2D backbone for 3D inference via PlaneCycle.

    Each block cycles through orthogonal planes (HW axial / DW coronal / DH
    sagittal) in cyclic order. The backbone name is auto-detected from
    its attributes; see _detect_backbone.

    Args:
        backbone: Pretrained 2D backbone (weights are not modified).
        cycle_order: Ordered plane labels cycled round-robin across blocks.
        pool_method: Global token pooling, 'PCg' adaptive avg (recommended) or 'PCm' mean.
    """

    def __init__(
            self,
            backbone,
            cycle_order: Tuple[str, ...] = ("HW", "DW", "DH", "HW"),
            pool_method: Literal["PCg", "PCm"] = "PCg",
            pool_D: bool = False,
            last_stage_disable_PC: bool = False,
    ) -> None:
        super().__init__()

        for p in cycle_order:
            if p not in PLANE_TO_AXES:
                raise ValueError(f"Unknown plane '{p}'. Choose from {list(PLANE_TO_AXES)}.")

        self.backbone = backbone
        self.backbone_name = _detect_backbone(backbone)
        self.cycle_order = cycle_order
        self.norm = backbone.norm
        self.pool_D = pool_D
        if self.backbone_name == 'dinov3':
            self.backbone.blocks = nn.ModuleList([
                PlaneCycleOp(block=blk, blk_idx=i, n_blocks=len(backbone.blocks), backbone_name='dinov3',
                             rope_embed=self.backbone.rope_embed,
                             cycle_order=cycle_order, pool_method=pool_method)
                for i, blk in enumerate(self.backbone.blocks)
            ])
            self.g_len = backbone.n_storage_tokens + 1  # CLS + storage tokens

        elif self.backbone_name == 'convnext':
            n_blocks = sum(len(s) for s in backbone.stages)
            blk_idx = 0
            for stage in backbone.stages:
                for i, block in enumerate(stage):
                    stage[i] = PlaneCycleOp(
                        block=block,
                        blk_idx=blk_idx,
                        n_blocks=n_blocks,
                        backbone_name='convnext',
                        cycle_order=cycle_order,
                        pool_method="",
                    )
                    blk_idx += 1
        else:
            raise ValueError(f"Unsupported backbone '{self.backbone_name}'. Choose from {SUPPORTED}.")

    def forward(self, x: Tensor):
        """
        Args:
            x: Input volume (B, C, D, H, W).
        Returns:
            xf: Spatial features (B, D, H, W, C).
            xcls: Pooled feature vector (B, D, C).
        """
        B, _C, D, _H, _W = x.shape

        if self.backbone_name == 'dinov3':
            xf, xg = self._vit_tokenise(x, B, D)
            for blk in self.backbone.blocks:
                xf, xg = blk(xf, xg)
            xf = self.norm(xf)  # (B, D, H, W, C)
            xcls = self.norm(xg[:, :, 0])  # (B, D, C)
            return xf, xcls

        elif self.backbone_name == 'convnext':
            x = x.permute(0, 2, 3, 4, 1)  # (B, C, D, H, W) → (B, D, H, W, C)
            for i in range(4):
                x = x.permute(0, 1, 4, 2, 3).flatten(0, 1)  # → (B*D, C, H, W)
                x = self.backbone.downsample_layers[i](x)  # → (B*D, C, H, W)
                x = x.permute(0, 2, 3, 1).unflatten(0, (B, D))  # → (B, D, H, W, C)
                if self.pool_D:
                    if i == 0:
                        D = D // 4
                    else:
                        D = D // 2
                    x = adaptive_avg_pool_along_dim(x, D, dim= 1)
                x = self.backbone.stages[i](x)
            xf = self.norm(x)  # (B, D, H, W, C)
            xcls = self.norm(x.mean(dim=[2, 3]))  # spatial mean → (B, D, C)
            return xf, xcls
        else:
            raise ValueError(f"Unsupported backbone '{self.backbone_name}'. Choose from {SUPPORTED}.")

    def _vit_tokenise(self, x: Tensor, B: int, D: int) -> Tuple[Tensor, Tensor]:
        """Slice-wise patch embedding, split into spatial xf and global xg."""
        x = x.flatten(0, 1)  # (B*D, C, H, W)
        x, (H, W) = self.backbone.prepare_tokens_with_masks(x)  # (B*D, g_len+H*W, C)
        C = x.shape[-1]
        xf = x[:, self.g_len:].reshape(B, D, H, W, C)  # (B, D, H, W, C)
        xg = x[:, :self.g_len].reshape(B, D, self.g_len, C)  # (B, D, g_len, C)
        return xf, xg
