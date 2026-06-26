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

## Requirements
Our code is built on top of the DINOv3 framework. 
1. DINOv3 Environment: Follow the [installation guide](https://github.com/facebookresearch/dinov3) to set up the basic dependencies.
2. Additional Packages: medmnist, transformers

All experiments are conducted on a single NVIDIA H200 GPU (141GB memory). 

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
from planecycle.converters.converter import PlaneCycleConverter

x = torch.randn(2, 3, 64, 256, 256) # （N, 3, D, H, W）

# Load a DINOv3 ViT backbone pretrained on web images
model = torch.hub.load(REPO_DIR, "dinov3_vits16", source="local", weights=<CHECKPOINT/URL/OR/PATH>)

# Convert the 2D backbone into a 3D PlaneCycle model
converter = PlaneCycleConverter(cycle_order=('HW', 'DW', 'DH', 'HW'), pool_method="PCg")
model = converter(model)

out = model(x)
```

## Code Structure

- `planecycle/`  
  Core implementation of the PlaneCycle framework.

  - `operators/` – Implementation of the PlaneCycle operators.
  - `converters/` – Converters for adapting pretrained ViT backbones.

- `models/`

  - `vision_transformer/` – Modified Vision Transformer implementation used in this project.

- `experiments/`  
  Scripts for running experiments and reproducing results.

  - `medmnist/` – Training and benchmarking pipelines for six 3D MedMNIST+ datasets.

## Results on MedMNIST+ (ConvNeXt Backbones)

AUC / ACC on six MedMNIST+ 3D classification datasets. Input (64³) is upsampled to 64×128×128 before processing. All models use pretrained weights. **Bold** = best within each model scale (S/B/L) per metric.

### Linear Probing

| Model | Method | Organ | Nodule | Fracture | Adrenal | Vessel | Synapse | **AVG** |
|-------|--------|-------|--------|----------|---------|--------|---------|---------|
| ConvNeXt-S | Slice2D | 98.2 / 80.5 | 85.7 / 82.3 | 60.3 / 43.8 | 84.4 / **82.9** | 85.0 / 88.2 | **85.8** / **83.0** | 83.2 / 76.8 |
| | ACSConv | **99.2** / 85.9 | 81.6 / 81.6 | 60.3 / 43.3 | 75.5 / 78.2 | 82.1 / 89.0 | 70.4 / 74.7 | 78.2 / 75.5 |
| | PC (Ours) | 99.1 / **86.9** | **86.4** / **82.6** | **61.1** / **44.6** | **86.8** / **82.9** | **86.9** / **90.0** | 82.7 / 82.1 | **83.8** / **78.2** |
| ConvNeXt-B | Slice2D | 97.8 / 79.2 | 82.8 / 78.1 | **66.7** / 48.8 | 78.7 / 79.5 | 84.4 / 88.7 | **87.0** / **83.8** | 82.9 / 76.3 |
| | ACSConv | 98.7 / 83.0 | 86.6 / **83.9** | 64.7 / **50.4** | 82.3 / 80.2 | 83.2 / **90.0** | 74.4 / 79.0 | 81.7 / 77.7 |
| | PC (Ours) | **99.3** / **87.4** | **87.9** / 83.2 | 57.0 / 43.8 | **90.2** / **85.2** | **85.3** / 88.2 | 82.8 / 80.7 | **83.7** / **78.1** |
| ConvNeXt-L | Slice2D | 98.2 / 80.3 | 87.2 / **87.4** | **64.0** / 43.8 | 77.5 / 79.9 | 81.2 / **89.8** | **86.3** / **83.0** | 82.4 / 77.3 |
| | ACSConv | **99.5** / **89.0** | **87.9** / 83.5 | 63.1 / **50.0** | 84.4 / 81.5 | 82.3 / 88.0 | 78.0 / 79.8 | 82.5 / **78.6** |
| | PC (Ours) | 99.4 / 87.9 | 82.7 / 80.0 | 58.4 / 43.8 | **87.7** / **83.9** | **89.0** / **89.8** | 84.4 / 80.1 | **83.6** / 77.6 |

### Full Fine-Tuning

| Model | Method | Organ | Nodule | Fracture | Adrenal | Vessel | Synapse | **AVG** |
|-------|--------|-------|--------|----------|---------|--------|---------|---------|
| ConvNeXt-S | Slice2D | 98.5 / 85.9 | 87.3 / 84.5 | 54.2 / 40.0 | 79.5 / 77.8 | 81.8 / 91.6 | 91.6 / 84.1 | 82.2 / 77.3 |
| | ACSConv | **99.9** / **96.1** | **91.5** / **87.1** | **69.7** / **51.7** | 66.8 / 76.8 | **94.9** / 91.9 | **96.2** / 90.9 | 86.5 / 82.4 |
| | PC (Ours) | **99.9** / 93.9 | 90.7 / 84.8 | 68.1 / 48.3 | **91.1** / **86.6** | 85.8 / **93.5** | 96.0 / **91.5** | **88.6** / **83.1** |
| ConvNeXt-B | Slice2D | 97.9 / 83.6 | 89.4 / 86.5 | 70.5 / 50.0 | 77.6 / 76.2 | 89.5 / 93.2 | 95.5 / 90.1 | 86.7 / 79.9 |
| | ACSConv | **100.0** / **97.4** | 92.2 / 86.5 | 57.1 / 37.5 | **87.9** / **85.6** | **96.5** / **94.5** | **97.0** / 91.5 | 88.4 / 82.2 |
| | PC (Ours) | 99.9 / 96.9 | **94.3** / **87.1** | **73.3** / **60.0** | 84.6 / 77.2 | 90.1 / 91.1 | 96.5 / **92.9** | **89.8** / **84.2** |
| ConvNeXt-L | Slice2D | 97.8 / 78.8 | **93.9** / 86.5 | 66.6 / 45.8 | 75.5 / 78.9 | 82.7 / 91.4 | 94.8 / 92.0 | 85.2 / 78.9 |
| | ACSConv | **100.0** / **97.5** | 89.4 / **87.7** | 61.8 / 37.5 | **89.4** / 85.6 | 82.8 / 86.4 | 97.2 / 93.2 | 86.8 / 81.3 |
| | PC (Ours) | 99.9 / 95.2 | 93.6 / **87.7** | **68.1** / **48.3** | **89.4** / **85.9** | **96.5** / **93.5** | **98.4** / **95.7** | **91.0** / **84.4** |

## How to run the experiments
* six 3D classification datasets(./experiments/medmnist/train_eval.py)
