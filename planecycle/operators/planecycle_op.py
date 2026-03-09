from typing import Callable, Tuple, Literal

import torch
from torch import Tensor, nn

from planecycle.operators.utils import adaptive_avg_pool_along_dim


class PlaneCycleOp(nn.Module):
    def __init__(self, pool_method: Literal["PCg", "PCm"] = "PCg"):
        """
        Args:
            pool_method:
                - "PCg": Use adaptive pooling to preserve token structure between slices.
                - "PCm": Use mean to get global volumetric representation.
        """
        super().__init__()
        if pool_method not in ("PCg", "PCm"):
            raise ValueError(f"pool_method must be 'PCg' or 'PCm', got {pool_method!r}")
        self.pool_method = pool_method

    def forward(self, x: Tensor, g: Tensor, f_layer: Callable[[Tensor], Tensor], plane_dim: int = 1, ) -> Tuple[
        Tensor, Tensor]:
        """Process 3D features and global tokens.
        Args:
            x: Features (B, D, H, W, C).
            g: Global tokens (B, D, g, C).
            f_layer: Pretrained 2D layer.
            plane_dim: Spatial axis for plane (1=D, 2=H, 3=W).
        Returns:
            x: Processed features (B, D, H, W, C).
            g: Processed tokens (B, D, g, C).
        """
        B, D, H, W, C = x.shape
        P = x.shape[plane_dim]
        g_len = g.size(2)

        # Step 1: reshape x to per-plane sequences and pool global tokens.
        x_plane = x.movedim(plane_dim, 1)
        x_seq = x_plane.reshape(B * P, -1, C)           # (B*P, L, C)

        if self.pool_method == "PCm":
            g = g.mean(dim=1, keepdim=True).expand(-1, P, -1, -1)
        else:
            g = adaptive_avg_pool_along_dim(g, output_size=P, dim=1)
        g_seq = g.reshape(B * P, g_len, C)             # (B*P, g_len, C)

        # Step 2: apply 2D layer
        t = f_layer(torch.cat([g_seq, x_seq], dim=1))

        # Step 3: restore 3D features
        x = t[:, g_len:].reshape_as(x_plane).movedim(1, plane_dim)
        g = t[:, :g_len].reshape(B, P, g_len, C)
        g = adaptive_avg_pool_along_dim(g, output_size=D, dim=1)

        return x, g
