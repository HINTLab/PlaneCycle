"""
PlaneCycle converter – unified interface for ViT and CNN backbones.

The 2D backbone is kept intact and unmodified.

Usage:
    backbone = torch.hub.load('models', 'dinov3_vits16', source='local', pretrained=True)
    model_3d = PlaneCycleConverter(backbone)
    out = model_3d(torch.randn(2, 3, 8, 224, 224)) # (B, C, D, H, W)
"""

from typing import List, Literal, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor

from planecycle.operators.planecycle_op import PLANE_TO_AXES, PlaneCycleOp
from planecycle.operators.utils import adaptive_avg_pool_along_dim

# Supported backbone types and names.
# backbone type: coarse category ('vit' / 'cnn')
# name: specific architecture, determines block iteration and forward details
#
#   vit/dinov3 – RoPE positional encoding, storage tokens (DINOv3)
#   cnn/convnext – iterate backbone.stages
#   cnn/resnet – iterate backbone.layer1 … layer4
SUPPORTED = {
    'vit': ('dinov3',),
    'cnn': ('convnext', 'resnet'),
}


def _detect_backbone(backbone: nn.Module) -> Tuple[str, str]:
    """Auto-detect (backbone_type, backbone_name) from backbone attributes.

    Pass either argument explicitly to override detection.
    """
    if hasattr(backbone, 'blocks') and hasattr(backbone, 'rope_embed'):
        return 'vit', 'dinov3'
    if hasattr(backbone, 'stages'):
        return 'cnn', 'convnext'
    if hasattr(backbone, 'layer1'):
        return 'cnn', 'resnet'
    raise ValueError(
        "Cannot auto-detect backbone. "
        "Pass backbone_type and backbone_name explicitly."
    )


class PlaneCycleConverter(nn.Module):
    """Wraps a 2D backbone for 3D inference via PlaneCycle.

    Each block cycles through orthogonal planes (HW axial / DW coronal / DH sagittal)
    in cycle order. Backbone type and name are auto-detected from attributes.

    Args:
        backbone: Pretrained 2D backbone (unmodified).
        cycle_order: Plane labels per block, cycled round-robin.
        pool_method: Global token pooling for ViT. 'PCg' adaptive (recommended), 'PCm' mean.
    """

    def __init__(
            self,
            backbone: nn.Module,
            cycle_order: Tuple[str, ...] = ("HW", "DW", "DH"),
            pool_method: Literal["PCg", "PCm"] = "PCg",
    ) -> None:
        super().__init__()

        for p in cycle_order:
            if p not in PLANE_TO_AXES:
                raise ValueError(f"Unknown plane '{p}'. Choose from {list(PLANE_TO_AXES)}.")

        self.backbone = backbone
        self.backbone_type, self.backbone_name = _detect_backbone(backbone)
        self.cycle_order = cycle_order

        if self.backbone_type == 'vit':
            if self.backbone_name == 'dinov3':
                self.planecycleop = PlaneCycleOp(pool_method=pool_method)
                self.g_len = backbone.n_storage_tokens + 1  # CLS + storage tokens
        elif self.backbone_type == 'cnn':
            raise NotImplementedError("CNN support is not yet implemented.")
        else:
            raise ValueError(f"Unknown backbone_type '{self.backbone_type}'. "
                             f"Choose from {list(SUPPORTED)}.")

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args: x: (B, C, D, H, W)
        Returns:
            xf: spatial features (B, D, H, W, C)
            xcls: CLS token per slice (B, P_last, C)
        """
        B, _C, D, _H, _W = x.shape

        if self.backbone_type == 'vit':
            xf, xg, H, W = self._vit_tokenise(x, B, D)
            if self.backbone_name == 'dinov3':
                for blk_idx, blk in enumerate(self.backbone.blocks):
                    plane = self.cycle_order[blk_idx % len(self.cycle_order)]
                    xf, xg = self.planecycleop(xf, xg, plane, blk, self.backbone.rope_embed)

                xf = self.backbone.norm(xf) # (B, D, H, W, C)
                xcls = self.backbone.norm(xg[:,:,0]) # (B, P_last, g_len, C) -> (B, P_last,C)
                return xf, xcls

        elif self.backbone_type == 'cnn':
            # TODO: implement CNN PlaneCycle.
            # convnext: iterate backbone.stages
            # resnet: iterate backbone.layer1 … layer4
            # No tokenisation, no global tokens, no RoPE. block(x) instead of block(tokens, rope).
            raise NotImplementedError

    # ── segmentation helper (ViT only) ────────────────────────────────────────

    def get_intermediate_layers(
            self,
            x: Tensor,
            n: Union[int, Sequence[int]] = 1,
            reshape: bool = False,
            return_class_token: bool = False,
            return_extra_tokens: bool = False,
            norm: bool = True,
    ) -> Tuple:
        """Return features from intermediate blocks (for segmentation heads).

        Args:
            x: (B, C, D, H, W)
            n: int → last n blocks; list → specific block indices.
            reshape: reshape patch tokens (B*D, h*w, C) → (B*D, C, h, w).
            return_class_token: also return CLS token per block.
            return_extra_tokens: also return storage tokens per block.
            norm: apply backbone LayerNorm before returning.

        Returns:
            Tuple per collected block: patch / (patch, cls) / (patch, extra) / (patch, cls, extra).
        """
        B, _C, D, _H, _W = x.shape
        xf, xg, H, W = self._vit_tokenise(x, B, D)

        total = len(self.backbone.blocks)
        blocks_to_take = (
            set(range(total - n, total)) if isinstance(n, int) else set(n)
        )

        collected: List[Tensor] = []
        for blk_idx, blk in enumerate(self.backbone.blocks):
            plane = self.cycle_order[blk_idx % len(self.cycle_order)]
            xf, xg = self.planecycleop(xf, xg, plane, blk, self.backbone.rope_embed)
            if blk_idx in blocks_to_take:
                collected.append(self._vit_merge_tokens(xf, xg, D))

        if norm:
            collected = [self.backbone.norm(o) for o in collected]

        cls_tokens = [o[:, 0] for o in collected]  # (B*D, C)
        extra_tokens = [o[:, 1:self.g_len] for o in collected]  # (B*D, n_storage, C)
        patch_tokens = [o[:, self.g_len:] for o in collected]  # (B*D, h*w, C)

        if reshape:
            patch_tokens = [
                t.reshape(B * D, H, W, -1).permute(0, 3, 1, 2).contiguous()
                for t in patch_tokens
            ]  # (B*D, C, h, w)

        if return_class_token and return_extra_tokens:
            return tuple(zip(patch_tokens, cls_tokens, extra_tokens))
        if return_class_token:
            return tuple(zip(patch_tokens, cls_tokens))
        if return_extra_tokens:
            return tuple(zip(patch_tokens, extra_tokens))
        return tuple(patch_tokens)

    # ── ViT helpers ───────────────────────────────────────────────────────────

    def _vit_tokenise(self, x: Tensor, B: int, D: int) -> Tuple[Tensor, Tensor, int, int]:
        """Slice-wise patch embedding → split into spatial xf and global xg."""
        x = x.permute(0, 2, 1, 3, 4).flatten(0, 1)  # (B*D, C, H, W)
        x, (H, W) = self.backbone.prepare_tokens_with_masks(x)  # x shape: (B*D, g_len+h*w, C)
        C = x.shape[-1]
        xf = x[:, self.g_len:].reshape(B, D, H, W, C)  # (B, D, H, W, C)
        xg = x[:, :self.g_len].reshape(B, D, self.g_len, C)  # (B, D, g_len, C)
        return xf, xg, H, W

    def _vit_merge_tokens(self, xf: Tensor, xg: Tensor, D: int) -> Tensor:
        """Pool xg to D slices and concatenate with xf → flat token sequence."""
        xg = adaptive_avg_pool_along_dim(xg, output_size=D, dim=1)  # (B, D, g_len, C)
        return torch.cat([
            xg.flatten(0, 1),  # (B*D, g_len, C)
            xf.flatten(0, 1).flatten(1, 2),  # (B*D, h*w, C)
        ], dim=1)  # (B*D, g_len+h*w, C)

