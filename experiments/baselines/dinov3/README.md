# DINOv3 baselines

Comparison methods from the paper, built on the official DINOv3 implementation
included under `dinov3/` at the repository root. All share the input/output
signature of `planecycle_converter`: a volume `(B, C, D, H, W)` in, spatial
features `xf` and pooled tokens `xcls` out.

**These are direct implementations, not converter-based.** Each class inherits
from the corresponding DINOv3 model and overrides `forward`; the method is
picked by a `block_type` argument. Parameter names are unchanged in every mode,
so the official checkpoints load with `load_state_dict(..., strict=True)`.

`BaselineViT` / `BaselineConvNeXt` also accept `block_type="PlaneCycle"`, so the
paper's sweeps can run every method through one class. An equivalence test
checks it against the converter — but **for release use, PlaneCycle should go
through `planecycle_converter`**, not this.

## Layout

```
baselines/dinov3/
├── vit/
│   ├── vision_transformer.py   BaselineViT — Slice2D / Flatten3D / PlaneCycle
│   └── universal_rope.py       1D/2D/3D UniversalRopePositionEmbedding (Flatten3D)
└── convnext/
    ├── convnext.py             BaselineConvNeXt — Slice2D / PlaneCycle
    └── convnext_acs.py         ACSConvNeXt — ACS convolution baseline (standalone)
```

`TriSlice` (2.5D) is `Slice2D` plus neighbouring-slice input channels, assembled
in `experiments/medmnist/loaders.py`, so it has no class of its own here.

## `vit/`

**`BaselineViT`** inherits from `DinoVisionTransformer`.
`BLOCK_TYPES = ("Slice2D", "Flatten3D", "PlaneCycle")`:

| `block_type` | What runs | `xf` | `xcls` |
|---|---|---|---|
| `"Slice2D"` | the unmodified 2D ViT on every axial slice | `(B, D, H', W', C)` | `(B, D, C)` — one CLS per slice |
| `"Flatten3D"` | one token sequence over the whole volume; 2D RoPE replaced by a 3D universal RoPE (`universal_rope.py`) | `(B, D, H', W', C)` | `(B, C)` — one CLS per volume |
| `"PlaneCycle"` | mirror of `planecycle.converter.ViTConverter`, driving the released `PlaneCycleViTOp` | `(B, D, H', W', C)` | `(B, P, C)` — one CLS per slice of the last plane |

## `convnext/`

**`BaselineConvNeXt`** inherits from the DINOv3 `ConvNeXt`.
`BLOCK_TYPES = ("Slice2D", "PlaneCycle")`:

| `block_type` | What runs | `xf` | `xcls` |
|---|---|---|---|
| `"Slice2D"` | the unmodified 2D ConvNeXt on every axial slice (D folded into batch once, end to end) | `(B, D, H', W', C)` | `(B, D, C)` — spatial mean per slice |
| `"PlaneCycle"` | mirror of `planecycle.converter.ConvNeXtConverter`, driving the released `PlaneCycleConvOp` | `(B, D, H', W', C)` | `(B, D, C)` |

`get_intermediate_layers` returns one entry per stage, each in the same format
as `forward`, and downsampling between stages stays axial: D is never reduced. Note
that ConvNeXt reduces H and W by 32×, so a 64×64 slice leaves a 2×2 feature map
before the per-slice mean; `--upsample_scale 2` gives 4×4 instead, at a
substantial memory cost.

### `convnext_acs.py` — ACS convolutions

The ACS baseline (Yang et al., *Reinventing 2D Convolutions for 3D Images*)
splits the 2D kernel bank channel-wise across the axial/coronal/sagittal
directions and runs natively in 5D. It still loads 2D pretrained weights (kernel
shapes are unchanged), and needs the external `acsconv` and `timm` packages (see
"Baseline dependencies" in the repository README). It is a full model of its own
because ACS changes the modules themselves, whereas the `block_type` modes only
change how data flows through unchanged 2D modules.
