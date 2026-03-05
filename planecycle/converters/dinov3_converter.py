from typing import Optional, Tuple, Type, Literal

import torch
import torch.nn as nn
from torch import Tensor

from planecycle.operators.planecycle_op import PlaneCycleOp
from planecycle.operators.utils import adaptive_avg_pool_along_dim


class PlaneCycleConverter:
    """Convert backbone blocks by wrapping them."""

    def __init__(
            self,
            keep_original: bool = False,
            blocks_attr: str = "blocks",
            cycle_order: Tuple[str, ...] = ('HW', 'DW', 'DH', 'HW'),
            pool_method: Literal["PCg", "PCm"] = "PCg",
    ):
        self.keep_original = keep_original
        self.blocks_attr = blocks_attr
        self.cycle_order = cycle_order
        self.pool_method = pool_method
        self._orig_attr = f"_{blocks_attr}_original"

    def __call__(self, backbone: nn.Module) -> nn.Module:
        """Convert backbone blocks."""
        blocks = getattr(backbone, self.blocks_attr)

        if not isinstance(blocks, nn.ModuleList):
            raise TypeError(f"Expected ModuleList, got {type(blocks).__name__}")
        if len(blocks) == 0:
            raise ValueError("blocks is empty")

        # Cache original
        if self.keep_original:
            setattr(backbone, self._orig_attr, blocks)

        # Get config
        n_storage_tokens = getattr(backbone, "n_storage_tokens", 0)
        rope_embed = getattr(backbone, "rope_embed", None)

        # Wrap blocks
        wrapped = nn.ModuleList([
            PlaneCycleBlock(
                blk2d=blk,
                rope_embed=rope_embed,
                n_storage_tokens=n_storage_tokens,
                block_idx=i,
                cycle_order=self.cycle_order,
                pool_method=self.pool_method,
            )
            for i, blk in enumerate(blocks)
        ])

        setattr(backbone, self.blocks_attr, wrapped)
        return backbone

    def restore(self, backbone: nn.Module) -> nn.Module:
        """Restore original blocks."""
        if not hasattr(backbone, self._orig_attr):
            raise RuntimeError("No cached original blocks")
        setattr(backbone, self.blocks_attr, getattr(backbone, self._orig_attr))
        return backbone


class PlaneCycleBlock(nn.Module):
    """Apply 2D block along different spatial planes.

    Input/Output: (BD, g_len + H*W, C)
    """
    PLANE_DIM_MAP = {"HW": 1, "DW": 2, "DH": 3}
    PLANE_SHAPE_MAP = {"HW": (2, 3), "DW": (1, 3), "DH": (1, 2)}

    def __init__(self, blk2d, rope_embed, n_storage_tokens,
                 block_idx, cycle_order=('HW', 'DW', 'DH', 'HW'),
                 pool_method: Literal["PCg", "PCm"] = "PCg"):
        super().__init__()
        self.blk2d = blk2d
        self.rope_embed = rope_embed
        self.cycle_order = cycle_order
        self.g_len = n_storage_tokens + 1
        self.block_idx = block_idx
        self.planecycleop = PlaneCycleOp(pool_method=pool_method)

    @property
    def plane(self) -> str:
        """Get current plane by block index."""
        return self.cycle_order[self.block_idx % len(self.cycle_order)]

    @property
    def plane_dim(self) -> int:
        """Get plane dimension."""
        return self.PLANE_DIM_MAP[self.plane]

    def _get_rope(self, shape: Tuple) -> Optional[Tensor]:
        """Get rope encoding for current plane."""
        if self.rope_embed is None:
            return None

        idx_0, idx_1 = self.PLANE_SHAPE_MAP[self.plane]
        return self.rope_embed(H=shape[idx_0], W=shape[idx_1])

    def forward(self, x: Tensor, shape: Tuple) -> Tensor:
        """Process tokens through spatial plane.
        Args:
            x: (BD, g_len + H*W, C)   shape: (B, D, H, W, C)
        Returns:
            (BD, g_len + H*W, C)
        """
        B, D, H, W, C = shape

        # Unpack tokens
        feature_maps = x[:, self.g_len:, :].reshape(B, D, H, W, C)
        glob_tokens = x[:, :self.g_len, :].reshape(B, D, self.g_len, C)

        # Apply processing
        x, g = self.planecycleop(
            feature_maps, glob_tokens,
            lambda t: self.blk2d(t, self._get_rope(shape)),
            plane_dim=self.plane_dim
        )

        # Pack output
        g = adaptive_avg_pool_along_dim(g, output_size=D, dim=1).flatten(0, 1)
        x = x.flatten(0, 1).flatten(1, 2)

        return torch.cat([g, x], dim=1)
