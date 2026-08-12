# 3D/2D/1D universal RoPE used by the Flatten3D baseline
# (extracted from the DINOv3 rope file so that `dinov3/` stays unmodified).

import math
from typing import Literal, Tuple

import numpy as np
import torch
from torch import Tensor, nn

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