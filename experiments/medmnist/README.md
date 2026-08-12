# MedMNIST Experiments

Linear Probing (LP) and Fine-Tuning (FT) of DINOv3 backbones on the
MedMNIST 3D datasets, comparing PlaneCycle against 2D / 2.5D / 3D lifting
baselines.

**Two ways to run:**

- **Single experiment** — one `train_eval.py` command, no Slurm and no extra
  config files. See the [Quick Start](#quick-start) below.
- **Full paper sweeps** (methods × datasets × seeds, as Slurm array jobs) —
  the YAML-driven launcher in [`scripts/`](scripts/README.md): copy two
  `.example` config files, then `./submit.sh planecycle/lp`;
  `analyze_wandb.py` collects the result tables back from W&B.

## Files

| File | Contents |
|------|----------|
| `train_eval.py` | Entry point: data pipeline, train/test loops, scheduler, W&B logging, CLI |
| `loaders.py` | Model assembly: build backbone + classifier per CLI args, load pretrained weights (`strict=True`) |
| `classifiers.py` | End-to-end classification models (backbone + slice aggregation + configurable head) |
| `transforms.py` | 3D input transforms and tensor utilities (`Lifted2DTransform`, `Native3DTransform`, `make_tri_slice`, `upsample_hw`) |
| `config.py` | Pretrained-weight filenames and backbone constructor kwargs per architecture |

The script adds the repository root to `sys.path` itself — run it directly
from this directory, no `PYTHONPATH` needed.

## Methods (`--block_type`)

| `--block_type` | Method |
|---|---|
| `PlaneCycle` | cycles each block across the HW / DW / DH planes |
| `Slice2D` | unmodified 2D model on every axial slice |
| `TriSlice` | 2.5D: Slice2D with (d-1, d, d+1) neighbouring-slice input channels |
| `Flatten3D` | one token sequence over the whole volume with a 3D RoPE (ViT only) |
| `ACS` | ACS convolutions, natively 3D (ConvNeXt only) |

`PlaneCycle` runs through `planecycle_converter`; the comparison methods are in
[`experiments/baselines/dinov3/`](../baselines/dinov3/README.md), which
documents each one. `--disable_converter` makes `PlaneCycle` use the
equivalence-tested baseline class instead of the converter (debug/ablation).

## Quick Start

Backbone weights must be downloaded from the DINOv3 repository (license-gated)
and placed in `--weight_dir`; expected filenames are in
`config.py → MODEL_WEIGHTS_MAP`. To reproduce the paper results, sweep
`--block_type` over the methods above.

Runs are logged to Weights & Biases (`--entity`). To try things out without a
W&B account, prefix any command with `WANDB_MODE=offline` — everything is then
kept local.

The commands below were run on a single NVIDIA H200 (141GB). Support for
[AutoDL](https://www.autodl.com/) instances is planned for a future release.

**Pool Method**: `--pool_method="PCg"` is used by default. You can also try
`PCm`, but note that `PCg` performs better than `PCm` in Linear Probing,
while they perform similarly in Fine-Tuning.

**Final Pooling**: `--final_pool_method="learn_to_pool"` (default) fuses the
per-slice features into one vector with a single linear layer over the slice
axis — `nn.Linear(D, 1)`, i.e. one learned weight per slice position, shared
across channels. `"mean"` averages the slices instead, with no learnable
parameters.

**Classification head**: a single `nn.Linear(embed_dim, n_classes)` in all
settings.

**Cycle Order**: The default plane traversal order for PlaneCycle is
`--cycle_order "HW" "DW" "DH" "HW"` as reported in the paper. You can
customize this to any order, e.g. `HW DW DH` or `HW DH DW`. We observe that
different plane orders yield slight performance variations across datasets.

### Linear Probing (LP)

```bash
python train_eval.py \
    --weight_dir="/path/to/weights" \
    --entity="your-wandb-entity" \
    --project_name="dinov3_lp_baseline" \
    --data_flag="nodulemnist3d" \
    --arch="dinov3_vits16" \
    --block_type="PlaneCycle" \
    --pool_method="PCg" \
    --final_pool_method="learn_to_pool" \
    --batch_size=32 \
    --num_epochs=200 \
    --num_workers=4 \
    --scheduler="WarmupCosineAnnealingLR" \
    --max_lr=1e-3 \
    --weight_decay=1e-5 \
    --warmup_epochs=10 \
    --output_root="/path/to/outputs" \
    --seed=42 \
    --training_method="LP" \
    --cycle_order "HW" "DW" "DH" "HW" \
    --download
```

### Fine-Tuning (FT)

Same as above with `--training_method="FT"`, fewer epochs, a lower learning rate
and stronger weight decay:

```bash
python train_eval.py \
    --weight_dir="/path/to/weights" \
    --entity="your-wandb-entity" \
    --project_name="dinov3_ft_baseline" \
    --data_flag="nodulemnist3d" \
    --arch="dinov3_vits16" \
    --block_type="PlaneCycle" \
    --pool_method="PCg" \
    --final_pool_method="learn_to_pool" \
    --batch_size=32 \
    --num_epochs=100 \
    --num_workers=4 \
    --scheduler="WarmupCosineAnnealingLR" \
    --max_lr=5e-5 \
    --weight_decay=0.05 \
    --warmup_epochs=10 \
    --output_root="/path/to/outputs" \
    --seed=42 \
    --training_method="FT" \
    --cycle_order "HW" "DW" "DH" "HW" \
    --download
```

## Training Parameters

### Essential

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--data_flag` | organmnist3d | Dataset: organmnist3d, nodulemnist3d, adrenalmnist3d, fracturemnist3d, vesselmnist3d, synapsemnist3d |
| `--arch` | dinov3_vits16 | Backbone: any ViT in `config.py → VIT_ARCHS` (vits16 … vit7b16) or dinov3_convnext_{tiny,small,base,large} |
| `--training_method` | LP | LP (freeze backbone, train head) or FT (train everything) |
| `--batch_size` | 32 | Batch size for training and evaluation |
| `--num_epochs` | 100 | Number of training epochs |
| `--seed` | 42 | Random seed |
| `--output_root` | ./outputs | Output directory |

### Method & Pooling

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--block_type` | PlaneCycle | PlaneCycle, Slice2D, TriSlice, Flatten3D, ACS (see Methods above) |
| `--cycle_order` | HW DW DH HW | Plane traversal order for PlaneCycle |
| `--pool_method` | PCg | PlaneCycle global-token pooling: PCg or PCm (ViT only) |
| `--final_pool_method` | learn_to_pool | Slice aggregation: learn_to_pool or mean |
| `--D_slices` | 64 | Number of depth slices seen by learn_to_pool (match the input D) |
| `--disable_converter` | – | PlaneCycle only: use the Baseline classes instead of `planecycle_converter` (debug/ablation) |
| `--channels_data` | repeated | Input channel encoding: repeated grayscale or neighbouring slices; TriSlice implies neighbors |

### Optimization

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--max_lr` | 1e-3 | Initial learning rate |
| `--min_lr` | 1e-6 | Minimum learning rate for cosine schedulers |
| `--scheduler` | WarmupCosineAnnealingLR | MultiStepLR, CosineAnnealingLR, or WarmupCosineAnnealingLR |
| `--warmup_epochs` | 10 | Warmup epochs (WarmupCosineAnnealingLR) |
| `--weight_decay` | 1e-2 | Weight decay for AdamW |

### Data & I/O

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--size` | 64 | Original dataset image size: 28 or 64 |
| `--target_resolution` | 64 | Spatial resolution after preprocessing |
| `--upsample_scale` | 1 | Upsample H/W before the backbone (D unchanged) |
| `--as_rgb` | – | Let the dataset repeat 1 channel to 3 |
| `--num_workers` | 4 | DataLoader workers |
| `--gpu_ids` | 0 | GPU id to run on (single-GPU training) |
| `--download` | – | Auto-download MedMNIST data |
| `--weight_dir` | – | Directory holding the pretrained backbone weights |
| `--model_path` | – | Optional checkpoint to evaluate / warm-start from |

### Logging

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--entity` | – | W&B entity (required for logging) |
| `--project_name` | dinov3 | W&B project name |
| `--run_name` | – | Custom W&B run name |
| `--run` | model1 | Suffix used by MedMNIST evaluator output files |

## Recommended Settings

| Setting | LP | FT |
|---------|----|----|
| `--num_epochs` | 200 | 100 |
| `--max_lr` | 1e-3 | 5e-5 |
| `--weight_decay` | 1e-5 | 0.05 |

## Notes

- Results are logged to Weights & Biases.
- `--model_family spectre|ctfm` switches to the SPECTRE / CT-FM comparison
  backbones used in the paper. Both need their own pretrained checkpoints and
  extra packages that are not in the base environment — see "Baseline
  dependencies" in the repository README.
