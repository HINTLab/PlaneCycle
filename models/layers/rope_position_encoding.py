# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import math
from typing import Literal, Tuple

import numpy as np
import torch
from torch import Tensor, nn


# RoPE positional embedding with no mixing of coordinates (axial) and no learnable weights
# Supports two parametrizations of the rope parameters: either using `base` or `min_period` and `max_period`.
class RopePositionEmbedding(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        *,
        num_heads: int,
        base: float | None = 100.0,
        min_period: float | None = None,
        max_period: float | None = None,
        normalize_coords: Literal["min", "max", "separate"] = "separate",
        shift_coords: float | None = None,
        jitter_coords: float | None = None,
        rescale_coords: float | None = None,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ):
        super().__init__()
        assert embed_dim % (4 * num_heads) == 0
        both_periods = min_period is not None and max_period is not None
        if (base is None and not both_periods) or (base is not None and both_periods):
            raise ValueError("Either `base` or `min_period`+`max_period` must be provided.")

        D_head = embed_dim // num_heads
        self.base = base
        self.min_period = min_period
        self.max_period = max_period
        self.D_head = D_head
        self.normalize_coords = normalize_coords
        self.shift_coords = shift_coords
        self.jitter_coords = jitter_coords
        self.rescale_coords = rescale_coords

        # Needs persistent=True because we do teacher.load_state_dict(student.state_dict()) to initialize the teacher
        self.dtype = dtype  # Don't rely on self.periods.dtype
        self.register_buffer(
            "periods",
            torch.empty(D_head // 4, device=device, dtype=dtype),
            persistent=True,
        )
        self._init_weights()

    def forward(self, *, H: int, W: int) -> tuple[Tensor, Tensor]:
        device = self.periods.device
        dtype = self.dtype
        dd = {"device": device, "dtype": dtype}

        # Prepare coords in range [-1, +1]
        if self.normalize_coords == "max":
            max_HW = max(H, W)
            coords_h = torch.arange(0.5, H, **dd) / max_HW  # [H]
            coords_w = torch.arange(0.5, W, **dd) / max_HW  # [W]
        elif self.normalize_coords == "min":
            min_HW = min(H, W)
            coords_h = torch.arange(0.5, H, **dd) / min_HW  # [H]
            coords_w = torch.arange(0.5, W, **dd) / min_HW  # [W]
        elif self.normalize_coords == "separate":
            coords_h = torch.arange(0.5, H, **dd) / H  # [H]
            coords_w = torch.arange(0.5, W, **dd) / W  # [W]
        else:
            raise ValueError(f"Unknown normalize_coords: {self.normalize_coords}")
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"), dim=-1)  # [H, W, 2]
        coords = coords.flatten(0, 1)  # [HW, 2]
        coords = 2.0 * coords - 1.0  # Shift range [0, 1] to [-1, +1]

        # Shift coords by adding a uniform value in [-shift, shift]
        if self.training and self.shift_coords is not None:
            shift_hw = torch.empty(2, **dd).uniform_(-self.shift_coords, self.shift_coords)
            coords += shift_hw[None, :]

        # Jitter coords by multiplying the range [-1, 1] by a log-uniform value in [1/jitter, jitter]
        if self.training and self.jitter_coords is not None:
            jitter_max = np.log(self.jitter_coords)
            jitter_min = -jitter_max
            jitter_hw = torch.empty(2, **dd).uniform_(jitter_min, jitter_max).exp()
            coords *= jitter_hw[None, :]

        # Rescale coords by multiplying the range [-1, 1] by a log-uniform value in [1/rescale, rescale]
        if self.training and self.rescale_coords is not None:
            rescale_max = np.log(self.rescale_coords)
            rescale_min = -rescale_max
            rescale_hw = torch.empty(1, **dd).uniform_(rescale_min, rescale_max).exp()
            coords *= rescale_hw

        # Prepare angles and sin/cos
        angles = 2 * math.pi * coords[:, :, None] / self.periods[None, None, :]  # [HW, 2, D//4]
        angles = angles.flatten(1, 2)  # [HW, D//2]
        angles = angles.tile(2)  # [HW, D]
        cos = torch.cos(angles)  # [HW, D]
        sin = torch.sin(angles)  # [HW, D]

        return (sin, cos)  # 2 * [HW, D]

    def _init_weights(self):
        device = self.periods.device
        dtype = self.dtype
        if self.base is not None:
            periods = self.base ** (
                2 * torch.arange(self.D_head // 4, device=device, dtype=dtype) / (self.D_head // 2)
            )  # [D//4]
        else:
            base = self.max_period / self.min_period
            exponents = torch.linspace(0, 1, self.D_head // 4, device=device, dtype=dtype)  # [D//4] range [0, 1]
            periods = base**exponents  # range [1, max_period / min_period]
            periods = periods / base  # range [min_period / max_period, 1]
            periods = periods * self.max_period  # range [min_period, max_period]
        self.periods.data = periods


# Universal RoPE supporting 1D, 2D, and 3D
class UniversalRopePositionEmbedding(nn.Module):
    def __init__(
            self,
            embed_dim: int,
            *,
            num_heads: int,
            sections: Tuple = (8, 12, 12),  # Default for 3D, sum must be D_head // 2
            base: float | None = 100.0,
            min_period: float | None = None,
            max_period: float | None = None,
            normalize_coords: Literal["min", "max", "separate"] = "separate",
            shift_coords: float | None = None,
            jitter_coords: float | None = None,
            rescale_coords: float | None = None,
            dtype: torch.dtype | None = None,
            device: torch.device | None = None,
    ):
        super().__init__()
        D_head = embed_dim // num_heads
        self.D_head = D_head
        self.sections = sections
        self.ndim = len(sections)

        # RoPE pairs dimensions: sum of section lengths must be half of D_head
        if sum(sections) != D_head // 2:
            raise ValueError(f"Sum of sections {sum(sections)} must be D_head // 2 ({D_head // 2})")

        both_periods = min_period is not None and max_period is not None
        if (base is None and not both_periods) or (base is not None and both_periods):
            raise ValueError("Either `base` or `min_period`+`max_period` must be provided.")

        self.base = base
        self.min_period = min_period
        self.max_period = max_period
        self.normalize_coords = normalize_coords
        self.shift_coords = shift_coords
        self.jitter_coords = jitter_coords
        self.rescale_coords = rescale_coords
        self.dtype = dtype

        # Register periods as a buffer
        self.register_buffer(
            "periods",
            torch.empty(sum(sections), device=device, dtype=dtype),
            persistent=False,
        )
        self._init_weights()

    def _init_weights(self):
        """Initializes periods for each section independently so each dimension
        gets a full spectrum of frequencies from high to low."""
        device = self.periods.device
        dtype = self.dtype
        all_periods = []

        for sec_len in self.sections:
            if self.base is not None:
                # Standard log-linear frequency scaling for each dimension
                p = self.base ** (
                        2 * torch.arange(sec_len, device=device, dtype=dtype) / (2 * sec_len)
                )
            else:
                # Min/Max period scaling
                base_ratio = self.max_period / self.min_period
                exponents = torch.linspace(0, 1, sec_len, device=device, dtype=dtype)
                p = (base_ratio ** exponents) / base_ratio * self.max_period
            all_periods.append(p)

        # Combine all sections into a single buffer
        self.periods.data = torch.cat(all_periods)

    def forward(self, *dims: int, **kwargs: int) -> Tuple[Tensor, Tensor]:
        """
        Args:
            *dims: Variable number of dimension sizes (e.g., L for 1D; H, W for 2D; D, H, W for 3D)
            sequence num
        Returns:
            sin, cos: Tensors of shape [N_tokens, D_head]
        """
        if kwargs:
            dims = tuple(kwargs.get(k, v) for k, v in kwargs.items())

        if len(dims) != self.ndim:
            raise ValueError(f"Expected {self.ndim} dimensions, but got {len(dims)}")

        device = self.periods.device
        dtype = self.dtype
        dd = {"device": device, "dtype": dtype}

        # 1. Coordinate Normalization
        coords_list = []
        if self.normalize_coords == "separate":
            for d in dims:
                coords_list.append(torch.arange(0.5, d, **dd) / d)
        else:
            ref_val = max(dims) if self.normalize_coords == "max" else min(dims)
            for d in dims:
                coords_list.append(torch.arange(0.5, d, **dd) / ref_val)

        # 2. Grid Generation & Transformation
        grid = torch.meshgrid(*coords_list, indexing="ij")
        coords = torch.stack(grid, dim=-1)  # [*dims, ndim]
        coords = coords.flatten(0, self.ndim - 1)  # [N_tokens, ndim]
        coords = 2.0 * coords - 1.0  # Map [0, 1] -> [-1, 1]

        # 3. Data Augmentation (Training only)
        if self.training:
            if self.shift_coords is not None:
                shift = torch.empty(self.ndim, **dd).uniform_(-self.shift_coords, self.shift_coords)
                coords = coords + shift[None, :]

            if self.jitter_coords is not None:
                jitter_max = np.log(self.jitter_coords)
                jitter = torch.empty(self.ndim, **dd).uniform_(-jitter_max, jitter_max).exp()
                coords = coords * jitter[None, :]

            if self.rescale_coords is not None:
                rescale_max = np.log(self.rescale_coords)
                rescale = torch.empty(1, **dd).uniform_(-rescale_max, rescale_max).exp()
                coords = coords * rescale

        # 4. Calculate Angles per Section
        angles_list = []
        curr_idx = 0
        for i, sec_len in enumerate(self.sections):
            # Extract the period set for this specific dimension
            p = self.periods[curr_idx: curr_idx + sec_len]
            # Broadcasting: [N, 1] / [1, sec_len] -> [N, sec_len]
            dim_angles = 2 * math.pi * coords[:, i, None] / p[None, :]
            angles_list.append(dim_angles)
            curr_idx += sec_len

        # Concatenate back to [N, D_head // 2]
        angles = torch.cat(angles_list, dim=-1)

        # 5. Final Sin/Cos with RoPE interleaving (tiling for axial symmetry)
        angles = angles.tile(2)  # [N, D_head]
        cos = torch.cos(angles)
        sin = torch.sin(angles)

        return (sin, cos)