"""
PlaneCycle operator modules – thin ``nn.Module`` wrappers over the pure
functions in ``functional.py``.

Follows the ``torch.nn`` convention: each module only stores its
configuration (wrapped block, plane, pooling method) in ``__init__`` and
its ``forward`` is a one-line delegation to the functional implementation.

"""

from typing import Optional

import torch.nn as nn
from torch import Tensor

from planecycle import functional as pcF
from planecycle.functional import PLANE_TO_AXES, Plane, PoolMethod


class PlaneCycleViTOp(nn.Module):
    """Apply a pretrained 2D transformer block to one plane of a 3D volume.

    See :func:`planecycle.functional.plane_cycle_vit` for the
    algorithm (Reshape -> Apply -> Reshape).

    Args:
        block: The wrapped 2D transformer block, called as ``block(tokens, rope)``.
        plane: Plane to slice along ('HW', 'DW' or 'DH').
        rope_embed: RoPE module from the ViT backbone (dinov3 only).
        pool_method: Global token pooling, 'PCg' adaptive avg (recommended) or 'PCm' mean.
    """

    def __init__(
        self,
        block: nn.Module,
        plane: Plane = "HW",
        rope_embed: Optional[nn.Module] = None,
        pool_method: PoolMethod = "PCg",
    ) -> None:
        super().__init__()
        if plane not in PLANE_TO_AXES:
            raise ValueError(
                f"Unknown plane {plane!r}. Choose from {list(PLANE_TO_AXES)}."
            )
        if pool_method not in ("PCg", "PCm"):
            raise ValueError(f"pool_method must be 'PCg' or 'PCm', got {pool_method!r}")
        self.block = block
        self.plane = plane
        self.rope_embed = rope_embed
        self.pool_method = pool_method

    def forward(self, xf: Tensor, xg: Tensor):
        return pcF.plane_cycle_vit(
            xf, xg, self.block, self.plane, self.rope_embed, self.pool_method
        )

    def extra_repr(self) -> str:
        return f"plane={self.plane!r}, pool_method={self.pool_method!r}"


class PlaneCycleConvOp(nn.Module):
    """Apply a pretrained 2D conv block to one plane of a 3D volume.

    See :func:`planecycle.functional.plane_cycle_conv2d` for the
    algorithm (Reshape -> Apply -> Reshape).

    Args:
        block: The wrapped 2D conv block, channels-first (N, C, S1, S2) in and out.
        plane: Plane to slice along ('HW', 'DW' or 'DH').
    """

    def __init__(self, block: nn.Module, plane: Plane = "HW") -> None:
        super().__init__()
        if plane not in PLANE_TO_AXES:
            raise ValueError(
                f"Unknown plane {plane!r}. Choose from {list(PLANE_TO_AXES)}."
            )
        self.block = block
        self.plane = plane

    def forward(self, xf: Tensor) -> Tensor:
        return pcF.plane_cycle_conv2d(xf, self.block, self.plane)

    def extra_repr(self) -> str:
        return f"plane={self.plane!r}"
