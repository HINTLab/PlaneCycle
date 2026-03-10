# MedMNIST

Training script for DINOv3 models on MedMNIST datasets using Linear Probing and Fine-Tuning.

# Quick Start

To reproduce all results, simply change `--block_type` between `PlaneCycle`, `Slice2D`, and `Flatten3D` in the commands below.

**Pool Method**: `--pool_method="PCg"` is used by default. You can also try `PCm`, but note that `PCg` performs better than `PCm` in Linear Probing, while they perform similarly in Fine-Tuning.

**Final Pooling**: We use a simple learnable fusion layer (`--final_pool_method="learn_to_pool"`) to fuse features across different slices. You can also use `"mean"` for simple averaging without a learnable fusion layer.

**3D RoPE**: `Flatten3D` uses an improved 3D Rotary Position Embedding (RoPE) that extends DINOv3's original 2D RoPE to 3D. See `models/layers/rope_position_encoding.py` for implementation details.

**Cycle Order**: The default plane traversal order for PlaneCycle is `--cycle_order "HW" "DW" "DH" "HW"` as reported in the paper. 
You can customize this to any order, e.g., `('HW', 'DW', 'DH')` or `('HW', 'DH', 'DW')`, or even define different planes for each block. 
We observe that different plane orders yield slight performance variations across different datasets.

### Linear Probing (LP)

```bash
python train_val.py \
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

```bash
python train_val.py \
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

### Essential Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--data_flag` | organmnist3d | Dataset: organmnist3d, nodulemnist3d, adrenalmnist3d, fracturemnist3d, vesselmnist3d, synapsemnist3d |
| `--arch` | dinov3_vits16 | Model architecture: dinov3_vits16, dinov3_vitb16, dinov3_vitl16 |
| `--batch_size` | 32 | Batch size for training |
| `--num_epochs` | 200 | Number of training epochs |
| `--seed` | 42 | Random seed |
| `--output_root` | ./outputs | Output directory |

### Learning Rate & Optimization

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--max_lr` | 1e-3 | Maximum learning rate |
| `--min_lr` | 1e-6 | Minimum learning rate |
| `--scheduler` | WarmupCosineAnnealingLR | Scheduler: MultiStepLR, CosineAnnealingLR, WarmupCosineAnnealingLR |
| `--warmup_epochs` | 10 | Warmup epochs (for WarmupCosineAnnealingLR) |
| `--weight_decay` | 1e-5 | Weight decay for AdamW |

### Architecture & Pooling

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--block_type` | PlaneCycle | Block type: PlaneCycle, Slice2D, Flatten3D |
| `--pool_method` | PCg | Pooling: PCg or PCm (for PlaneCycle) |
| `--final_pool_method` | learn_to_pool | Final pooling: mean, learn_to_pool, or no_pool |
| `--cycle_order` | HW DW DH HW | Plane order for PlaneCycle |
| `--D_slices` | 64 | Depth slices for pooling |
| `--concat_patch_token` | - | Add flag to concatenate patch tokens |

### Data & I/O

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--size` | 64 | Original image size |
| `--target_resolution` | 64 | Target resolution |
| `--as_rgb` | - | Add flag to convert to RGB |
| `--num_workers` | 0 | DataLoader workers |
| `--download` | - | Add flag to auto-download datasets |
| `--weight_dir` | - | Weights directory path |

### Logging & Checkpoints

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--entity` | <your_wandb_entity> | W&B entity name (required for logging) |
| `--project_name` | dinov3 | W&B project name |
| `--run_name` | - | Custom run name |
| `--model_path` | - | Path to pretrained checkpoint |
| `--training_method` | LP | LP (linear probe) or FT (finetune) |

## Recommended Settings

### Linear Probing vs Fine-Tuning

| Setting | LP | FT |
|---------|----|----|
| `--training_method` | LP | FT |
| `--num_epochs` | 200 | 100 |
| `--max_lr` | 1e-3 | 5e-5 |
| `--weight_decay` | 1e-5 | 0.05 |

## Notes
- Results are logged to Weights & Biases
- **Linear Probing (LP)**: Freezes the backbone, only trains the classification head
- **Fine-Tuning (FT)**: Updates all parameters, requires lower learning rate and higher weight decay