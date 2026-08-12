"""
Functional (stateless) implementation of the PlaneCycle operators.

Mirrors the ``torch.nn.functional`` convention: every function here is pure —
inputs, the wrapped 2D block, and all hyper-parameters are passed as
arguments, and nothing is stored between calls. The thin ``nn.Module``
wrappers in ``ops.py`` only remember the configuration and
delegate to these functions.

All spatial tensors are channels-last ``(B, D, H, W, C)``. A "plane" is a
pair of active axes; the remaining axis is the slice axis:

    HW (axial)    – slice along D, RoPE over H x W
    DW (coronal)  – slice along H, RoPE over D x W
    DH (sagittal) – slice along W, RoPE over D x H
"""

from typing import Dict, Literal, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

Plane = Literal["HW", "DW", "DH"]
PoolMethod = Literal["PCg", "PCm"]

# For each plane: (slice_axis, rope_row_axis, rope_col_axis) in (B, D, H, W, C).
PLANE_TO_AXES: Dict[str, Tuple[int, int, int]] = {
    "HW": (1, 2, 3),  # slice along D(1), RoPE over H(2) x W(3)
    "DW": (2, 1, 3),  # slice along H(2), RoPE over D(1) x W(3)
    "DH": (3, 1, 2),  # slice along W(3), RoPE over D(1) x H(2)
}


def adaptive_avg_pool_along_dim(x: Tensor, output_size: int, dim: int = 1) -> Tensor:
    """Adaptive average pool along dimension `dim` to `output_size`.

    Args:
        x: Input tensor.
        output_size: Target size for dimension `dim`.
        dim: Dimension to pool (supports negative indexing).
    """
    dim %= x.ndim
    if x.size(dim) == output_size:
        return x

    # Edge case: input_size == 1 mathematically means replicate the single slice
    # `output_size` times. CUDA's adaptive_avg_pool1d has a known illegal memory
    # access bug with input_size=1, output_size>1; use expand instead.
    if x.size(dim) == 1:
        shape = list(x.shape)
        shape[dim] = output_size
        return x.expand(shape).contiguous()

    # adaptive_avg_pool1d pools the last dim, so move `dim` there and back.
    x = torch.moveaxis(x, dim, -1)
    *batch_shape, last_dim = x.shape
    x = F.adaptive_avg_pool1d(x.reshape(-1, 1, last_dim), output_size)
    return torch.moveaxis(x.reshape(*batch_shape, output_size), -1, dim)


def fold_plane(x: Tensor, slice_dim: int) -> Tensor:
    """Fold a volume into a stack of 2D slices of the selected plane:
    (B, D, H, W, C) -> (B*P, S1, S2, C).

    Two steps: move the slice axis next to the batch axis, then merge them.
    ``P = x.size(slice_dim)`` and (S1, S2) are the two remaining spatial axes,
    so each of the B*P slices is a 2D image the wrapped block can consume.
    """
    return x.movedim(slice_dim, 1).flatten(0, 1)


def unfold_plane(x: Tensor, batch_size: int, slice_dim: int) -> Tensor:
    """Inverse of :func:`fold_plane`: (B*P, S1, S2, C) -> (B, D, H, W, C).

    Splits the batch axis back into (B, P), then moves the slice axis back to
    its original position.
    """
    return x.unflatten(0, (batch_size, -1)).movedim(1, slice_dim)


def pool_global_tokens(
    xg: Tensor, num_slices: int, method: PoolMethod = "PCg"
) -> Tensor:
    """Resample global (CLS + storage) tokens across slices: (B, P', g, C) -> (B, P, g, C).

    'PCg' adaptively average-pools P' -> P (recommended); 'PCm' collapses to
    the mean over P' and broadcasts it to all P slices.
    """
    if method == "PCm":
        return xg.mean(dim=1, keepdim=True).expand(-1, num_slices, -1, -1)
    if method == "PCg":
        return adaptive_avg_pool_along_dim(xg, output_size=num_slices, dim=1)
    raise ValueError(f"pool_method must be 'PCg' or 'PCm', got {method!r}")


def plane_cycle_vit(
    xf: Tensor,
    xg: Tensor,
    block,
    plane: Plane,
    rope_embed=None,
    pool_method: PoolMethod = "PCg",
) -> Tuple[Tensor, Tensor]:
    """Apply a pretrained 2D transformer block to one plane of a 3D volume.

    Args:
        xf: Spatial tokens (B, D, H, W, C).
        xg: Global tokens (B, P', g_len, C); P' is the previous plane's slice count.
        block: 2D transformer block, called as ``block(tokens, rope)``.
        plane: Which plane to slice along; see PLANE_TO_AXES.
        rope_embed: Optional RoPE module, called with the plane's (H, W).
        pool_method: How to resample global tokens to P slices; see pool_global_tokens.

    Returns:
        xf: (B, D, H, W, C), xg: (B, P, g_len, C) where P = xf.size(slice axis).
    """
    slice_dim, rope_row, rope_col = PLANE_TO_AXES[plane]
    B, C = xf.size(0), xf.size(-1)
    P, g_len = xf.size(slice_dim), xg.size(2)

    slices = fold_plane(xf, slice_dim)  # (B*P, S1, S2, C)
    S1, S2 = slices.size(1), slices.size(2)
    x_seq = slices.flatten(1, 2)  # (B*P, S1*S2, C)

    xg = pool_global_tokens(xg, P, pool_method)  # (B, P, g_len, C)
    g_seq = xg.flatten(0, 1)  # (B*P, g_len, C)

    rope = rope_embed(H=xf.size(rope_row), W=xf.size(rope_col)) if rope_embed else None
    tokens = block(torch.cat([g_seq, x_seq], dim=1), rope)  # (B*P, g_len+L, C)

    xf = unfold_plane(tokens[:, g_len:].unflatten(1, (S1, S2)), B, slice_dim)
    xg = tokens[:, :g_len].unflatten(0, (B, P))
    return xf, xg


def plane_cycle_conv2d(xf: Tensor, block, plane: Plane) -> Tensor:
    """Apply a pretrained 2D conv block to one plane of a 3D volume.

    Args:
        xf: Feature volume (B, D, H, W, C).
        block: 2D conv block taking and returning channels-first (N, C, S1, S2).
        plane: Which plane to slice along; see PLANE_TO_AXES.

    Returns:
        xf: (B, D, H, W, C).
    """
    slice_dim = PLANE_TO_AXES[plane][0]
    B = xf.size(0)

    slices = fold_plane(xf, slice_dim)  # (B*P, S1, S2, C)
    out = block(slices.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
    return unfold_plane(out, B, slice_dim)
