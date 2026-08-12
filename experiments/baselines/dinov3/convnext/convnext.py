"""
Baseline ConvNeXt: DINOv3 ConvNeXt lifted to a 3D-volume interface.

``BaselineConvNeXt`` inherits from the DINOv3 ``ConvNeXt`` — nothing is copied;
stem, stages, norms and weights are all reused unchanged. It adds a 5D ``forward``
((B, C, D, H, W) in) with two ``block_type`` modes:

  - "Slice2D":    every axial slice through the unmodified 2D ConvNeXt
  - "PlaneCycle": mirrors ``planecycle.converter.ConvNeXtConverter`` using the
                  released ``PlaneCycleConvOp`` (the math lives there only)

Both modes return xf (B, D, H', W', C) and xcls (B, D, C) (spatial mean per
slice). ``get_intermediate_layers`` matches ``planecycle.converter``: entries
are per *stage* (4 total), each as if the network stopped after that stage,
normed with the inherited per-stage ``norms`` table (Identity except the last).

State-dict guarantee: keys are identical to ``ConvNeXt`` in every mode (the
PlaneCycle ops are kept in plain lists, never registered), so pretrained
checkpoints load with ``strict=True`` regardless of ``block_type``.

For release use, PlaneCycle should go through ``planecycle_converter``; this
class exists so the paper's experiment configs can switch methods with one
flag against a uniform interface.
"""

from typing import List, Sequence, Tuple, Union

from torch import Tensor, nn

from dinov3.models.convnext import ConvNeXt
from planecycle.ops import PlaneCycleConvOp

BLOCK_TYPES = ("Slice2D", "PlaneCycle")


class BaselineConvNeXt(ConvNeXt):
    def __init__(
        self,
        *args,
        block_type: str = "Slice2D",
        cycle_order: Tuple[str, ...] = ("HW", "DW", "DH", "HW"),
        **kwargs,
    ) -> None:
        if block_type not in BLOCK_TYPES:
            raise ValueError(f"Unknown block_type {block_type!r}. Choose from {BLOCK_TYPES}.")
        super().__init__(*args, **kwargs)
        self.block_type = block_type

        if block_type == "PlaneCycle":
            self.cycle_order = cycle_order
            # Plain nested lists, never registered: state_dict keys stay those
            # of the base model. Ops are parameter-less wrappers holding
            # references to the registered stage blocks; the plane index runs
            # continuously across stages, as in ConvNeXtConverter.
            idx = 0
            self._pc_ops: List[List[PlaneCycleConvOp]] = []
            for stage in self.stages:
                ops = []
                for block in stage:
                    ops.append(
                        PlaneCycleConvOp(block=block, plane=cycle_order[idx % len(cycle_order)])
                    )
                    idx += 1
                self._pc_ops.append(ops)

    # ── forward (dispatch on block_type) ──────────────────────────────────────

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """x: (B, C, D, H, W) -> (xf (B, D, H', W', C), xcls (B, D, C))."""
        run = self._run_planecycle if self.block_type == "PlaneCycle" else self._run_slice2d
        x = run(x, upto=len(self.stages) - 1)
        return self.norm(x), self.norm(x.mean(dim=[2, 3]))

    def _run_slice2d(self, x: Tensor, upto: int, take=frozenset(), collected=None) -> Tensor:
        """Fold D into batch once, run the plain 2D network end-to-end, unfold
        once — exactly the 2D model's path, no per-stage round-trips. Unfolded
        volumes of the stages in ``take`` are appended to ``collected``
        (get_intermediate_layers only; forward passes neither)."""
        B, _C, D, _H, _W = x.shape
        x = x.permute(0, 2, 1, 3, 4).flatten(0, 1)  # (B*D, C, H, W)
        for i in range(upto + 1):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            if i in take:
                collected.append(x.permute(0, 2, 3, 1).unflatten(0, (B, D)))
        return x.permute(0, 2, 3, 1).unflatten(0, (B, D))  # (B, D, H', W', C)

    def _run_planecycle(self, x: Tensor, upto: int, take=frozenset(), collected=None) -> Tensor:
        """Plane-cycled stages on a channels-last volume (B, D, H, W, C); the
        fold happens per stage because downsampling is 2D (axial) while the
        plane ops need the full volume. Stages in ``take`` are appended to
        ``collected`` (get_intermediate_layers only; forward passes neither)."""
        x = x.permute(0, 2, 3, 4, 1)  # (B, C, D, H, W) -> (B, D, H, W, C)
        B, D = x.shape[:2]
        for i in range(upto + 1):
            x = x.permute(0, 1, 4, 2, 3).flatten(0, 1)  # -> (B*D, C, H, W)
            x = self.downsample_layers[i](x)
            x = x.permute(0, 2, 3, 1).unflatten(0, (B, D))  # (B, D, H, W, C)
            for op in self._pc_ops[i]:
                x = op(x)
            if i in take:
                collected.append(x)
        return x

    # ── intermediate layers (for segmentation decoders) ───────────────────────

    def get_intermediate_layers(
        self,
        x: Tensor,
        n: Union[int, Sequence[int]] = 1,
        return_class_token: bool = False,
        norm: bool = True,
    ) -> Tuple:
        """Per-stage features, same output format as ``planecycle.converter``:
        each entry is ``forward``'s output as if the network stopped after that
        stage, normed with the inherited per-stage ``norms`` table."""
        total = len(self.stages)
        take = set(range(total - n, total)) if isinstance(n, int) else set(n)
        if not all(0 <= i < total for i in take):
            raise ValueError(f"Stage indices out of range [0, {total}): {sorted(take)}")

        collected: list = []
        run = self._run_planecycle if self.block_type == "PlaneCycle" else self._run_slice2d
        run(x, upto=max(take), take=take, collected=collected)

        outputs = []
        for i, vol in zip(sorted(take), collected):
            norm_fn = self.norms[i] if norm else nn.Identity()
            feat = norm_fn(vol)
            outputs.append(
                (feat, norm_fn(vol.mean(dim=[2, 3]))) if return_class_token else feat
            )
        return tuple(outputs)
