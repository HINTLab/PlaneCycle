# planecycle

The PlaneCycle package: convert a pretrained 2D backbone (DINOv3 ViT or
ConvNeXt, auto-detected) into a model for 3D volumetric inference — no new
parameters, pretrained weights untouched. See the repository README for the
paper, results, setup and a usage example.

`model = planecycle_converter(backbone, ...)` maps a volume `(B, C, D, H, W)`
to `(xf, xcls)`: spatial features `(B, D, H, W, C)` (e.g. for segmentation)
and per-slice global tokens `(B, D, C)` under the default cycle order (in
general `(B, P, C)`, with P set by the last block's plane). The slice
axis of `xcls` is deliberately kept: in the paper's experiments, the
classification head first aggregates the slices with a learned pooling — a
linear layer over the slice axis (`learn_to_pool`) that weights informative
slices — with plain mean pooling as its special case. Downstream heads are
free to aggregate the slices in any other way that fits their task.

## Example

```python
import torch
from planecycle import planecycle_converter

x = torch.randn(2, 3, 64, 256, 256)  # (B, C, D, H, W)

# Load a DINOv3 ViT backbone pretrained on web images
backbone = torch.hub.load(REPO_DIR, "dinov3_vits16", source="local",
                          weights="<PATH/TO/CHECKPOINT>")

# Convert the 2D backbone into a 3D PlaneCycle model
model = planecycle_converter(backbone, cycle_order=("HW", "DW", "DH", "HW"),
                             pool_method="PCg")

xf, xcls = model(x)  # xf:   (B, D, H', W', C) spatial features; H', W' = feature
                     #       grid (input H, W over the backbone's downsampling)
                     # xcls: (B, D, C) per-slice global tokens
```

## Multi-layer features (for dense prediction)

```python
feats = model.get_intermediate_layers(x, n=4)                  # last 4 layers
feats = model.get_intermediate_layers(x, n=[3, 7, 11])         # specific layers
feats = model.get_intermediate_layers(x, n=4, return_class_token=True)
```

Each entry has exactly `forward`'s output format, as if the network stopped at
that layer. For ViT a "layer" is a transformer block; for ConvNeXt it is a
stage (4 total, with per-stage grid size and channels).

## Files

| File | Contents |
|---|---|
| `converter.py` | `planecycle_converter` factory + per-backbone converters (ViT, ConvNeXt) |
| `ops.py` | `PlaneCycleViTOp` / `PlaneCycleConvOp` — the plane-wise block operators (`nn.Module`) |
| `functional.py` | the stateless math: `plane_cycle_vit` / `plane_cycle_conv2d`, plane folding/unfolding, global-token pooling |

`ops.py`/`functional.py` follow the `torch.nn` / `torch.nn.functional`
split: modules hold configuration, functions hold the math.
