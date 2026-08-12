# PlaneCycle

PlaneCycle: Training-Free 2D-to-3D Lifting of Foundation Models Without Adapters ([arXiv](https://arxiv.org/abs/2603.04165))

[//]: # (> 🚀 **Coming soon:** A unified operator for ViTs and CNNs *&#40;tests already passed, full release on the way!&#41;*)

## Overview

<div align="center">
  <img src="assets/feature_visualization.png" width="800" alt="DINOv3 Feature Visualization">
  <p align="center" style="max-width: 800px; margin: 0 auto;">
    <i>
      <b>PCA visualizations of frozen lifted DINOv3 features.</b><br/>
      Evaluated on three 3D datasets across HW, DW, and DH planes (inconsistencies circled).
    </i>
  </p>
</div>

<br/>
<br/>

<div align="center">
  <img src="assets/Planecycle.png" width="800" alt="PlaneCycle Overview">
  <p align="center" style="max-width: 800px; margin: 0 auto;">
    <i>
      <b>Overview of PlaneCycle across three orthogonal planes (HW, DW, DH).</b><br/>
      Flattened slice tokens are processed by shared ViT layers with plane-specific RoPE.
    </i>
  </p>
</div>

<br/>

## Pretrained Models
We utilize DINOv3 as the backbone for our 2D-to-3D lifting. 
DINOv3 Weights: Please follow the official repository [facebookresearch/dinov3](https://github.com/facebookresearch/dinov3) to download the pretrained checkpoints.

## Environment setup

The code needs **Python 3.11** and PyTorch (2.10 / CUDA 12.8 in our runs). Pick
either conda or uv — both install everything in one command.

**conda** (creates an env named `planecycle`):

```bash
conda env create -f environment.yml
conda activate planecycle
```

**uv** (faster; PyPI wheels only — torch's wheel bundles CUDA):

```bash
uv venv --python 3.11 && source .venv/bin/activate
uv pip install -r requirements.txt
```

`environment.yml` / `requirements.txt` cover PlaneCycle itself and all
experiments (classification + segmentation), but **not** the comparison
baselines — see below.

### Baseline dependencies (optional)

The comparison baselines need extra packages that are not part of the base
environment. Install them only for the method(s) you want to reproduce:

```bash
pip install timm==1.0.25 huggingface-hub==0.36.0   # SPECTRE
pip install lighter-zoo==0.1.3                     # CT-FM
```

PlaneCycle / Slice2D / TriSlice / Flatten3D run on the base environment alone.
The ConvNeXt-only ACS baseline needs one more package — see
[`docs/RESULTS_CONVNEXT.md`](docs/RESULTS_CONVNEXT.md).

The ViT experiments are conducted on a single NVIDIA H200 GPU (141GB memory).
The ConvNeXt results below use 2× upsampled input and need more memory than an
H200 provides at batch size 32; they were run on a single NVIDIA B300 (288GB).

Memory requirements on MedMNIST datasets are detailed in the paper:
- Linear Probing: <10GB
- Fine-Tuning: 17GB (ViT-S), 33GB (ViT-B), 87GB (ViT-L)

Memory requirements on LIDC & MMWHS dataset:
- Linear Probing: 17GB (ViT-S), 18GB (ViT-B), 32GB (ViT-L)
- Fine-Tuning: 32GB (ViT-S), 44GB (ViT-B), 87GB (ViT-L)

## PlaneCycle Converter

Configure the following parameters in the converter.

### Key Parameters

- `cycle_order` Specifies the sequence of spatial planes for feature aggregation.
  Each block can be assigned a plane from `('HW', 'DW', 'DH').

  * Default (Paper): `('HW', 'DW', 'DH', 'HW')`
  * Alternatives: Support any order, e.g., `('HW', 'DW', 'DH')` or `('HW', 'DH', 'DW')`, or even define different planes for each block. 

- `pool_method`  
  Specifies how global tokens are aggregated across planes.

  - `"PCg"` (default): Uses adaptive pooling to preserve spatial token structure.
  - `"PCm"`: Uses mean pooling to obtain a global volumetric representation.

### Example

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

## Code Structure

**`planecycle/` is the method** — everything else is either the pretrained
backbone it wraps or code for reproducing the paper's experiments.

```
planecycle/            the PlaneCycle operator and converter (the contribution)
├── functional.py        plane reshaping and token pooling (pure functions)
├── ops.py               PlaneCycleViTOp / PlaneCycleConvOp (one 2D block, one plane)
└── converter.py         planecycle_converter: wraps a whole 2D backbone

dinov3/                official DINOv3 ViT / ConvNeXt, unmodified
hubconf.py             torch.hub entry point for the DINOv3 backbones

experiments/           reproduction code
├── medmnist/            six 3D MedMNIST+ classification datasets
├── segmentation/        LIDC and MMWHS segmentation
└── baselines/           comparison methods (Slice2D, Flatten3D, ACS, SPECTRE)
```

Applying PlaneCycle to your own model needs `planecycle/` only; see the example
above.

## Results

The paper's ViT-S/B/L results are in [`Paper`](https://arxiv.org/abs/2603.04165)
(Tables 1–3: linear probing, full fine-tuning, and segmentation).

PlaneCycle on the DINOv3 **ConvNeXt** backbones — follow-up work beyond the
paper — is reported separately in
[`docs/RESULTS_CONVNEXT.md`](docs/RESULTS_CONVNEXT.md).

## How to run the experiments

| Experiment | Entry point | Docs |
|---|---|---|
| MedMNIST+ 3D classification (six datasets) | `experiments/medmnist/train_eval.py` | [`experiments/medmnist/README.md`](experiments/medmnist/README.md) |
| Paper sweeps on Slurm (all methods × datasets × seeds) | `experiments/medmnist/scripts/submit.sh` | [`experiments/medmnist/scripts/README.md`](experiments/medmnist/scripts/README.md) |
| LIDC / MMWHS segmentation | `experiments/segmentation/seg_training.py` | [`experiments/segmentation/README.md`](experiments/segmentation/README.md) |
| Comparison baselines | selected via `--block_type` | [`experiments/baselines/dinov3/README.md`](experiments/baselines/dinov3/README.md) |
