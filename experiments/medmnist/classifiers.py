"""End-to-end classification models over the 3D-lifted backbones.

Every backbone (planecycle_converter, BaselineViT/ConvNeXt, SPECTRE, CT-FM)
maps a volume to features; each Classifier here owns a backbone, aggregates
its output and classifies. The classification head is a single linear layer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from transforms import make_tri_slice, upsample_hw


def build_head(embed_dim, n_classes):
    """Classification head: a single linear layer."""
    return nn.Linear(embed_dim, n_classes)


class Dinov3Classifier(nn.Module):
    """DINOv3-family backbone + slice aggregation + classification head."""

    def __init__(
        self,
        *,
        backbone,
        n_classes,
        embed_dim,
        final_pool_method: str = "learn_to_pool",
        upsample_scale=False,
        block_type="PlaneCycle",
        final_slices=64,
        channels_data: str = "repeated",
    ):
        super().__init__()
        self.backbone = backbone
        self.final_pool_method = final_pool_method
        if self.final_pool_method == "learn_to_pool":
            self.pool_slices = nn.Linear(final_slices, 1)
        self.head = build_head(embed_dim, n_classes)
        self.upsample_scale = upsample_scale
        self.block_type = block_type
        self.channels_data = channels_data

    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Input encoding: optional H/W upsampling, optional neighbour-slice
        channels. (block_type="TriSlice" is normalized to
        channels_data="neighbors" by the loader.)"""
        if self.upsample_scale > 1:
            x = upsample_hw(x, scale_factor=self.upsample_scale)  # → (B, C, D, 2H, 2W)
        if self.channels_data == "neighbors":
            x = make_tri_slice(x)
        return x

    def _pool(self, xcls: torch.Tensor) -> torch.Tensor:
        """Aggregate per-slice CLS tokens (B, D, C) → (B, C)."""
        if self.block_type == "Flatten3D":
            return xcls  # one CLS per volume, already (B, C)
        if self.final_pool_method == "learn_to_pool":
            return self.pool_slices(xcls.permute(0, 2, 1)).squeeze(-1)
        if self.final_pool_method == "mean":
            return xcls.mean(dim=1)
        raise ValueError(f"Unknown final_pool_method: {self.final_pool_method}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args: x: (B, 3, D, H, W)
        Returns: logits (B, n_classes)
        """
        x = self._preprocess(x)
        xf, xcls = self.backbone(x)
        xcls = self._pool(xcls)
        return self.head(xcls)


class SpectreClassifier(nn.Module):
    """SPECTRE backbone + classification head on the CLS token.

    Wraps a SPECTRE VisionTransformer (global_pool='') and handles the axis
    permutation from our (B, C, D, H, W) convention to SPECTRE's
    (B, C, H, W, D).
    """

    def __init__(self, *, backbone, n_classes, embed_dim):
        super().__init__()
        self.backbone = backbone
        self.head = build_head(embed_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, D, H, W) → (B, 1, H, W, D) for SPECTRE
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        features = self.backbone(x)  # (B, T+1, embed_dim)  global_pool=''
        cls = features[:, 0, :]  # CLS token → (B, embed_dim)
        return self.head(cls)


class CTFMClassifier(nn.Module):
    """CT-FM (SegResEncoder) backbone + classification head.

    Global-average-pools the deepest encoder feature map to a fixed-length
    embedding, then classifies. Input: (B, 1, D, H, W), values in [0, 1].
    """

    def __init__(self, *, backbone, n_classes, embed_dim):
        super().__init__()
        self.backbone = backbone
        self.head = build_head(embed_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, D, H, W)
        features = self.backbone(x)[-1]  # (B, C, d, h, w)
        cls = F.adaptive_avg_pool3d(features, 1).flatten(1)  # (B, C)
        return self.head(cls)
