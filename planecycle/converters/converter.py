"""
PlaneCycle converter – unified interface for ViT and CNN backbones.

The 2D backbone is kept intact and unmodified.

"""

from typing import Literal, Tuple

import torch.nn as nn
from torch import Tensor

from planecycle.operators.planecycle_op import PLANE_TO_AXES, PlaneCycleOp

# Supported backbone names and the attributes used to detect each:
#   vit – ViT with RoPE positional encoding and storage tokens
#   convnext – ConvNeXt; iterates backbone.stages
SUPPORTED = ("vit", "convnext")


def _detect_backbone(backbone: nn.Module) -> str:
    """Auto-detect a backbone name from its attributes."""
    if hasattr(backbone, "blocks") and hasattr(backbone, "rope_embed"):
        return "vit"
    if hasattr(backbone, "stages"):
        return "convnext"
    raise ValueError(f"Cannot auto-detect backbone. Supported: {SUPPORTED}.")


class PlaneCycleConverter(nn.Module):
    """Wraps a 2D backbone for 3D inference via PlaneCycle.

    当前支持基于dinov3预训练的vit和convnext各种变体，其他的模型结果还没有测试
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
    ) -> None:
        super().__init__()

        for p in cycle_order:
            if p not in PLANE_TO_AXES:
                raise ValueError(
                    f"Unknown plane '{p}'. Choose from {list(PLANE_TO_AXES)}."
                )

        self.backbone_name = _detect_backbone(backbone)
        self.backbone = backbone
        self.cycle_order = cycle_order
        self.norm = backbone.norm

        if self.backbone_name == "vit":
            self.backbone.blocks = nn.ModuleList(
                [
                    PlaneCycleOp(
                        backbone_name="vit",
                        block=blk,
                        rope_embed=self.backbone.rope_embed,
                        plane=cycle_order[i % len(cycle_order)],
                        pool_method=pool_method,
                    )
                    for i, blk in enumerate(self.backbone.blocks)
                ]
            )
            self.g_len = backbone.n_storage_tokens + 1  # CLS + storage tokens

        elif self.backbone_name == "convnext":
            blk_idx = 0
            for stage in backbone.stages:
                for i, block in enumerate(stage):
                    stage[i] = PlaneCycleOp(
                        block=block,
                        backbone_name="convnext",
                        plane=cycle_order[blk_idx % len(cycle_order)],
                        pool_method=None,
                    )
                    blk_idx += 1
        else:
            raise ValueError(
                f"Unsupported backbone '{self.backbone_name}'. Choose from {SUPPORTED}."
            )

    def forward(self, x: Tensor):
        """
        Args:
            x: Input volume (B, C, D, H, W).
        Returns:
            xf: Spatial features (B, D, H, W, C).
            xcls: feature vector (B, P, C) for vit and (B, D, C) for cnn.
        """
        B, _C, D, _H, _W = x.shape
        if self.backbone_name == "vit":
            # (B, C, D, H, W) ->(B, D, C, H, W)
            x = x.permute(0, 2, 1, 3, 4).flatten(0, 1)  # (B*D, C, H, W)
            # x (B*D, g_len+H*W, C)
            x, (H, W) = self.backbone.prepare_tokens_with_masks(x)
            C = x.shape[-1]
            xf = x[:, self.g_len :].reshape(B, D, H, W, C)  # (B, D, H, W, C)
            xg = x[:, : self.g_len].reshape(B, D, self.g_len, C)  # (B, D, g_len, C)

            for blk in self.backbone.blocks:
                xf, xg = blk(xf, xg)

            xf = self.norm(xf)  # (B, D, H, W, C)
            xcls = self.norm(xg[:, :, 0])  # (B, P, C)
            return xf, xcls

        elif self.backbone_name == "convnext":
            x = x.permute(0, 2, 3, 4, 1)  # (B, C, D, H, W) → (B, D, H, W, C)
            for i in range(4):
                x = x.permute(0, 1, 4, 2, 3).flatten(0, 1)  # → (B*D, C, H, W)
                x = self.backbone.downsample_layers[i](x)  # → (B*D, C, H, W)
                x = x.permute(0, 2, 3, 1).unflatten(0, (B, D))  # → (B, D, H, W, C)
                x = self.backbone.stages[i](x)
            xf = self.norm(x)  # (B, D, H, W, C)
            xcls = self.norm(x.mean(dim=[2, 3]))  # spatial mean → (B, D, C)
            return xf, xcls
        else:
            raise ValueError(
                f"Unsupported backbone '{self.backbone_name}'. Choose from {SUPPORTED}."
            )
