"""
PlaneCycle converter.

:func:`planecycle_converter` converts a pretrained 2D backbone into an
``nn.Module`` for 3D inference. Currently supported: DINOv3-pretrained ViT and
ConvNeXt (auto-detected); other pretrained foundation models (e.g. ResNet) will
be considered in future releases. Construction replaces each 2D block with a
PlaneCycle operator in place — the pretrained weights are never modified.
``forward`` takes a volume ``(B, C, D, H, W)`` and runs tokenize (stem /
patch-embed) -> the plane-cycled blocks -> finalize (norm + pool), returning a
plain tuple ``(xf, xcls)``: spatial features ``(B, D, H, W, C)`` and pooled
tokens ``(B, P, C)``. In output shapes, H and W denote the feature-grid size
(the input H, W divided by the ViT patch size or the ConvNeXt stage strides);
D is never downsampled.

``xcls`` deliberately keeps the per-slice axis P instead of averaging over it:
downstream heads can then aggregate slices as they see fit (e.g. a learned
linear pooling that weights informative slices; plain mean is recoverable as
its special case). P is the slice count of the *last* block's plane — with the
default cycle order ending in HW, P equals D, i.e. one token per axial slice.

ViT and ConvNeXt share the same PlaneCycle operator mechanism (fold the slice
axis into batch -> apply the 2D block on the selected plane -> restore); the
two converter classes exist only because the surrounding input/output glue
differs — token layout and RoPE for ViT vs stage-wise downsampling for
ConvNeXt. Supporting a new backbone = one new converter class + one dispatch
line in :func:`planecycle_converter`.
"""

from typing import Literal, Sequence, Tuple, Union

import torch.nn as nn
from torch import Tensor

from planecycle.functional import PLANE_TO_AXES
from planecycle.ops import PlaneCycleConvOp, PlaneCycleViTOp


class ViTConverter(nn.Module):
    """DINOv3 ViT: flat block list, RoPE per plane, CLS + storage tokens."""

    def __init__(self, backbone, cycle_order, pool_method) -> None:
        super().__init__()
        self.backbone = backbone
        backbone.blocks = nn.ModuleList(
            PlaneCycleViTOp(
                block=blk,
                plane=cycle_order[i % len(cycle_order)],  # round-robin cycling
                rope_embed=backbone.rope_embed,
                pool_method=pool_method,
            )
            for i, blk in enumerate(backbone.blocks)
        )
        self.g_len = backbone.n_storage_tokens + 1  # CLS + storage tokens

    def _tokenize(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Fold D into batch, patch-embed, split global vs patch tokens:
        (B, C, D, H, W) -> xf (B, D, H, W, C), xg (B, D, g, C)."""
        B, _C, D, _H, _W = x.shape
        x = x.permute(0, 2, 1, 3, 4).flatten(0, 1)  # (B*D, C, H, W)
        x, (H, W) = self.backbone.prepare_tokens_with_masks(x)
        C = x.shape[-1]
        xf = x[:, self.g_len :].reshape(B, D, H, W, C)
        xg = x[:, : self.g_len].reshape(B, D, self.g_len, C)
        return xf, xg

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        xf, xg = self._tokenize(x)
        # run: cycle the plane-wise blocks, threading patch and global tokens
        for blk in self.backbone.blocks:
            xf, xg = blk(xf, xg)
        # finalize: norm; per-slice CLS tokens, P axis kept for downstream pooling
        norm = self.backbone.norm
        return norm(xf), norm(xg[:, :, 0])  # xf (B,D,H,W,C), xcls (B,P,C)

    def get_intermediate_layers(
        self,
        x: Tensor,
        n: Union[int, Sequence[int]] = 1,
        return_class_token: bool = False,
        norm: bool = True,
    ) -> Tuple:
        """Features from intermediate blocks (e.g. for segmentation decoders).

        Each entry has exactly ``forward``'s output format, as if the network
        stopped at that block: xf (B, D, H, W, C) and, when requested, the
        per-slice CLS tokens (B, P, C). P is that block's plane slice count,
        so it varies across collected layers — entries are not meant to be
        stacked; aggregate per layer as needed.

        Args:
            x: (B, C, D, H, W)
            n: int -> last n blocks; list -> specific block indices.
            return_class_token: also return the CLS tokens per block.
            norm: apply the backbone's final LayerNorm.

        Returns:
            Tuple with one entry per collected block:
            xf, or (xf, cls) pairs.
        """
        total = len(self.backbone.blocks)
        take = set(range(total - n, total)) if isinstance(n, int) else set(n)
        if not all(0 <= i < total for i in take):
            raise ValueError(f"Block indices out of range [0, {total}): {sorted(take)}")

        xf, xg = self._tokenize(x)
        collected = []
        for i, blk in enumerate(self.backbone.blocks):
            xf, xg = blk(xf, xg)
            if i in take:
                collected.append((xf, xg))

        norm_fn = self.backbone.norm if norm else nn.Identity()
        return tuple(
            (norm_fn(xf_i), norm_fn(xg_i[:, :, 0]))
            if return_class_token
            else norm_fn(xf_i)
            for xf_i, xg_i in collected
        )


class ConvNeXtConverter(nn.Module):
    """ConvNeXt: stem + 4 stages, downsampling (axial-only) between stages, no
    global tokens. Downsampling stays plane-agnostic (folds D into batch)."""

    def __init__(self, backbone, cycle_order) -> None:
        super().__init__()
        self.backbone = backbone
        idx = 0
        for stage in backbone.stages:
            for i, block in enumerate(stage):
                stage[i] = PlaneCycleConvOp(
                    block=block,
                    plane=cycle_order[idx % len(cycle_order)],  # round-robin
                )
                idx += 1

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        # tokenize: to channels-last volume (B, D, H, W, C)
        x = x.permute(0, 2, 3, 4, 1)
        B, D = x.shape[:2]  # constant through run (only H, W get downsampled)
        # run: per stage, axial-fold to downsample H,W, then the plane-wise stage
        for downsample, stage in zip(
            self.backbone.downsample_layers, self.backbone.stages
        ):
            x = x.permute(0, 1, 4, 2, 3).flatten(0, 1)  # -> (B*D, C, H, W)
            x = downsample(x)  # 2D downsample of H, W
            x = x.permute(0, 2, 3, 1).unflatten(0, (B, D))  # -> (B, D, H, W, C)
            x = stage(x)
        # finalize: norm; spatial mean per slice (the CNN analogue of per-slice
        # CLS tokens), D axis kept for downstream pooling
        norm = self.backbone.norm
        return norm(x), norm(x.mean(dim=[2, 3]))  # xf (B,D,H,W,C), xcls (B,D,C)

    def get_intermediate_layers(
        self,
        x: Tensor,
        n: Union[int, Sequence[int]] = 1,
        return_class_token: bool = False,
        norm: bool = True,
    ) -> Tuple:
        """Features from intermediate stages (e.g. for segmentation decoders).

        Each entry has exactly ``forward``'s output format, as if the network
        stopped after that stage: xf (B, D, H_i, W_i, C_i) and, when requested,
        the per-slice pooled tokens (B, D, C_i). Grid size and channels differ
        per stage. Mirrors the original ConvNeXt: ``backbone.norms`` is applied
        per stage — a real LayerNorm on the last stage only, Identity earlier.

        Args:
            x: (B, C, D, H, W)
            n: int -> last n stages; list -> specific stage indices.
            return_class_token: also return the pooled tokens per stage.
            norm: apply the backbone's per-stage norm.

        Returns:
            Tuple with one entry per collected stage:
            xf, or (xf, pooled tokens) pairs.
        """
        total = len(self.backbone.stages)
        take = set(range(total - n, total)) if isinstance(n, int) else set(n)
        if not all(0 <= i < total for i in take):
            raise ValueError(f"Stage indices out of range [0, {total}): {sorted(take)}")

        x = x.permute(0, 2, 3, 4, 1)  # (B, D, H, W, C)
        B, D = x.shape[:2]
        outputs = []
        for i, (downsample, stage) in enumerate(
            zip(self.backbone.downsample_layers, self.backbone.stages)
        ):
            x = x.permute(0, 1, 4, 2, 3).flatten(0, 1)  # -> (B*D, C, H, W)
            x = downsample(x)
            x = x.permute(0, 2, 3, 1).unflatten(0, (B, D))  # -> (B, D, H, W, C)
            x = stage(x)
            if i in take:
                norm_fn = self.backbone.norms[i] if norm else nn.Identity()
                feat = norm_fn(x)
                outputs.append(
                    (feat, norm_fn(x.mean(dim=[2, 3]))) if return_class_token else feat
                )
        return tuple(outputs)


def planecycle_converter(
    backbone: nn.Module,
    cycle_order: Tuple[str, ...] = ("HW", "DW", "DH", "HW"),
    pool_method: Literal["PCg", "PCm"] = "PCg",
) -> nn.Module:
    """Convert a pretrained 2D backbone into a 3D PlaneCycle model.

    The backbone type is auto-detected from its attributes; pretrained weights
    are not modified. Currently supported: DINOv3-pretrained ViT and ConvNeXt;
    other pretrained foundation models will be considered in future releases.

    Args:
        backbone: Pretrained 2D backbone (ViT or ConvNeXt).
        cycle_order: Ordered plane labels cycled round-robin across blocks.
        pool_method: Global token pooling, 'PCg' adaptive avg (recommended) or
            'PCm' mean. Ignored by CNN backbones (no global tokens).

    Returns:
        An ``nn.Module`` whose ``forward(x)`` maps (B, C, D, H, W) -> (xf, xcls).
    """
    for p in cycle_order:
        if p not in PLANE_TO_AXES:
            raise ValueError(f"Unknown plane '{p}'. Choose from {list(PLANE_TO_AXES)}.")

    if hasattr(backbone, "blocks") and hasattr(backbone, "rope_embed"):
        return ViTConverter(backbone, cycle_order, pool_method)
    if hasattr(backbone, "stages"):
        return ConvNeXtConverter(backbone, cycle_order)
    raise ValueError(
        "Cannot auto-detect backbone. Supported: ViT (blocks + rope_embed), "
        "ConvNeXt (stages)."
    )
