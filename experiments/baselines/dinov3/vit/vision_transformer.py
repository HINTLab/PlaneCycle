"""
Baseline ViT: DINOv3 ViT lifted to a 3D-volume interface, one flag per method.

``BaselineViT`` inherits from ``DinoVisionTransformer`` — nothing is copied;
blocks, tokenizer, norm and weights are all reused unchanged. It adds a 5D
``forward`` ((B, C, D, H, W) in) with three ``block_type`` modes:

  - "Slice2D":    every axial slice through the unmodified 2D ViT
                  -> xf (B, D, H', W', C), xcls (B, D, C)
  - "Flatten3D":  one token sequence over the whole volume, attention under
                  a 3D universal RoPE (the 2D RoPE is bypassed, not replaced)
                  -> xf (B, D, H', W', C), xcls (B, C)
  - "PlaneCycle": mirrors ``planecycle.converter.ViTConverter`` using the
                  released ``PlaneCycleViTOp`` (the math lives there only)
                  -> xf (B, D, H', W', C), xcls (B, P, C)

``get_intermediate_layers`` (for segmentation decoders) matches
``planecycle.converter``: each entry has exactly ``forward``'s output format,
as if the network stopped at that block.

State-dict guarantee: keys are identical to ``DinoVisionTransformer`` in every
mode (the PlaneCycle ops are kept in a plain list, never registered), so
pretrained checkpoints load with ``strict=True`` regardless of ``block_type``.

For release use, PlaneCycle should go through ``planecycle_converter``; this
class exists so the paper's experiment configs can switch methods with one
flag against a uniform interface.
"""

from typing import List, Sequence, Tuple, Union

import torch
from torch import Tensor, nn

from experiments.baselines.dinov3.vit.universal_rope import UniversalRopePositionEmbedding
from dinov3.models.vision_transformer import DinoVisionTransformer, dtype_dict
from planecycle.ops import PlaneCycleViTOp

BLOCK_TYPES = ("Slice2D", "Flatten3D", "PlaneCycle")


class BaselineViT(DinoVisionTransformer):
    def __init__(
        self,
        *args,
        block_type: str = "Slice2D",
        cycle_order: Tuple[str, ...] = ("HW", "DW", "DH", "HW"),
        pool_method: str = "PCg",
        **kwargs,
    ) -> None:
        if block_type not in BLOCK_TYPES:
            raise ValueError(f"Unknown block_type {block_type!r}. Choose from {BLOCK_TYPES}.")
        super().__init__(*args, **kwargs)
        self.block_type = block_type

        if block_type == "Flatten3D":
            # 3D universal RoPE, registered ALONGSIDE the inherited rope_embed
            # (not replacing it): its buffers are persistent=False, so it adds
            # no state_dict keys, while rope_embed keeps its persistent buffer
            # -> keys stay identical to the base model.
            self.rope3d = UniversalRopePositionEmbedding(
                embed_dim=self.embed_dim,
                num_heads=kwargs.get("num_heads", 12),
                sections=(8, 12, 12),  # sums to head_dim // 2
                base=kwargs.get("pos_embed_rope_base", 100.0),
                min_period=kwargs.get("pos_embed_rope_min_period"),
                max_period=kwargs.get("pos_embed_rope_max_period"),
                normalize_coords=kwargs.get("pos_embed_rope_normalize_coords", "separate"),
                shift_coords=kwargs.get("pos_embed_rope_shift_coords"),
                jitter_coords=kwargs.get("pos_embed_rope_jitter_coords"),
                rescale_coords=kwargs.get("pos_embed_rope_rescale_coords", 2.0),
                dtype=dtype_dict[kwargs.get("pos_embed_rope_dtype", "fp32")],
                device=kwargs.get("device"),
            )
        elif block_type == "PlaneCycle":
            self.cycle_order = cycle_order
            self.pool_method = pool_method
            # Plain list, never registered: state_dict keys stay those of the
            # base model. The ops are parameter-less wrappers holding
            # references to self.blocks / self.rope_embed, so weights loaded
            # or moved after construction are seen through them automatically.
            self._pc_ops: List[PlaneCycleViTOp] = [
                PlaneCycleViTOp(
                    block=blk,
                    plane=cycle_order[i % len(cycle_order)],
                    rope_embed=self.rope_embed,
                    pool_method=pool_method,
                )
                for i, blk in enumerate(self.blocks)
            ]

    # ── shared helpers ────────────────────────────────────────────────────────

    @property
    def _g_len(self) -> int:
        return self.n_storage_tokens + 1  # CLS + storage tokens

    def _tokenize_slices(self, x: Tensor):
        """(B, C, D, H, W) -> per-slice tokens via the inherited 2D tokenizer:
        (B*D, g+H'*W', C), plus (B, D, H', W')."""
        B, _C, D, _H, _W = x.shape
        x = x.permute(0, 2, 1, 3, 4).flatten(0, 1)  # (B*D, C, H, W)
        tokens, (H, W) = self.prepare_tokens_with_masks(x)
        return tokens, B, D, H, W

    def _tokenize_volume(self, x: Tensor):
        """(B, C, D, H, W) -> one token sequence per volume (Flatten3D):
        (B, g+D*H'*W', C) with a single CLS (+ storage) per volume."""
        tokens, B, D, H, W = self._tokenize_slices(x)
        C = tokens.shape[-1]
        patches = tokens[:, self._g_len :].reshape(B, D * H * W, C)
        cls_token = (self.cls_token + 0 * self.mask_token).expand(B, -1, -1)
        globals_ = [cls_token]
        if self.n_storage_tokens > 0:
            globals_.append(self.storage_tokens.expand(B, -1, -1))
        tokens = torch.cat(globals_ + [patches], dim=1)
        return tokens, B, D, H, W

    def _split_planecycle(self, x: Tensor):
        """Tokenize and split into the (xf, xg) pair the PlaneCycle ops thread."""
        tokens, B, D, H, W = self._tokenize_slices(x)
        C = tokens.shape[-1]
        xf = tokens[:, self._g_len :].reshape(B, D, H, W, C)
        xg = tokens[:, : self._g_len].reshape(B, D, self._g_len, C)
        return xf, xg

    # ── forward (dispatch on block_type) ──────────────────────────────────────

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """x: (B, C, D, H, W) -> (xf, xcls); see module docstring per mode."""
        if self.block_type == "PlaneCycle":
            return self._forward_planecycle(x)
        if self.block_type == "Flatten3D":
            return self._forward_flatten3d(x)
        return self._forward_slice2d(x)

    def _forward_slice2d(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        tokens, B, D, H, W = self._tokenize_slices(x)
        rope = self.rope_embed(H=H, W=W) if self.rope_embed is not None else None
        for blk in self.blocks:
            tokens = blk(tokens, rope)
        tokens = self.norm(tokens)
        C = tokens.shape[-1]
        xf = tokens[:, self._g_len :].reshape(B, D, H, W, C)
        xcls = tokens[:, 0].reshape(B, D, C)  # one CLS per slice
        return xf, xcls

    def _forward_flatten3d(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        tokens, B, D, H, W = self._tokenize_volume(x)
        rope = self.rope3d(D=D, H=H, W=W)  # 3D universal RoPE
        for blk in self.blocks:
            tokens = blk(tokens, rope)
        tokens = self.norm(tokens)
        C = tokens.shape[-1]
        xf = tokens[:, self._g_len :].reshape(B, D, H, W, C)
        xcls = tokens[:, 0]  # one CLS per volume: (B, C)
        return xf, xcls

    def _forward_planecycle(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        # Mirrors planecycle.converter.ViTConverter.forward; the plane-cycling
        # math is the released PlaneCycleViTOp.
        xf, xg = self._split_planecycle(x)
        for op in self._pc_ops:
            xf, xg = op(xf, xg)
        return self.norm(xf), self.norm(xg[:, :, 0])  # xcls: (B, P, C)

    # ── intermediate layers (for segmentation decoders) ───────────────────────

    def get_intermediate_layers(
        self,
        x: Tensor,
        n: Union[int, Sequence[int]] = 1,
        return_class_token: bool = False,
        norm: bool = True,
    ) -> Tuple:
        """Features from intermediate blocks, same output format as
        ``planecycle.converter``: each entry has exactly ``forward``'s output
        format, as if the network stopped at that block.

        Args:
            x: (B, C, D, H, W)
            n: int -> last n blocks; list -> specific block indices.
            return_class_token: also return the per-block xcls (its shape
                follows the mode, see the module docstring; for PlaneCycle the
                P of each entry follows that block's plane).
            norm: apply the model's final LayerNorm.
        """
        total = len(self.blocks)
        take = set(range(total - n, total)) if isinstance(n, int) else set(n)
        if not all(0 <= i < total for i in take):
            raise ValueError(f"Block indices out of range [0, {total}): {sorted(take)}")
        norm_fn = self.norm if norm else nn.Identity()

        if self.block_type == "PlaneCycle":
            xf, xg = self._split_planecycle(x)
            collected = []
            for i, op in enumerate(self._pc_ops):
                xf, xg = op(xf, xg)
                if i in take:
                    collected.append((xf, xg))
            return tuple(
                (norm_fn(f), norm_fn(g[:, :, 0])) if return_class_token else norm_fn(f)
                for f, g in collected
            )

        # Slice2D / Flatten3D: one token sequence, one rope, plain block loop
        if self.block_type == "Flatten3D":
            tokens, B, D, H, W = self._tokenize_volume(x)
            rope = self.rope3d(D=D, H=H, W=W)
            split_cls = lambda t: t[:, 0]  # (B, C)
        else:
            tokens, B, D, H, W = self._tokenize_slices(x)
            rope = self.rope_embed(H=H, W=W) if self.rope_embed is not None else None
            split_cls = lambda t: t[:, 0].reshape(B, D, -1)  # (B, D, C)

        collected = []
        for i, blk in enumerate(self.blocks):
            tokens = blk(tokens, rope)
            if i in take:
                collected.append(tokens)

        outputs = []
        for t in collected:
            t = norm_fn(t)
            xf = t[:, self._g_len :].reshape(B, D, H, W, t.shape[-1])
            outputs.append((xf, split_cls(t)) if return_class_token else xf)
        return tuple(outputs)
