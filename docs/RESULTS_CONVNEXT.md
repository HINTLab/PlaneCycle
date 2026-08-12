# Results on MedMNIST+ (ConvNeXt backbones)

PlaneCycle applied to the DINOv3 **ConvNeXt** backbones. These experiments are
follow-up work beyond the paper, which reports ViT-S/B/L only.

Input volumes (64³) are upsampled to 64×128×128 before the backbone, since
ConvNeXt reduces H and W by 32× and a 64×64 slice would otherwise leave a 2×2
feature map. At batch size 32 this does not fit on a 141 GB GPU; these runs used
a single NVIDIA B300 (288 GB).

All models use pretrained weights. **Bold** = best within each model scale
(S/B/L) per metric. Values are AUC / ACC in percent.

The **ACSConv** rows are the ACS baseline (Yang et al., *Reinventing 2D
Convolutions for 3D Images*), a natively-3D ConvNeXt that still loads the 2D
pretrained weights. It is the only method here that needs extra packages:

```bash
pip install acsconv==0.1.1 timm==1.0.25
```

Run it with `--block_type ACS`, or through the
`baselines/convnext_acs_{lp,ft}` sweep configs.

## Linear Probing

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

## Full Fine-Tuning

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
