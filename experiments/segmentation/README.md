# Root
## `featdice_lidc.py` & `featdice_mmwhs.ipynb`

These are used to evaluate FeatDice scores mentioned in our paper.

## `seg_training.py`

This is used to train and test **PlaneCycle** on different configurations.

The backbone is built according to `backbone_model.block_type` in the cfg: `PlaneCycle` loads a plain 2D backbone via `torch.hub` and converts it with `planecycle_converter`; `Slice2D` / `Flatten3D` use `BaselineViT` from `experiments/baselines/dinov3/vit` (its state-dict keys equal the plain backbone's, so the official checkpoints load unchanged).

## `cfgs`

This folder contains full fine-tuning and linear probing example files of configuration.

Notably, `block_type` sits at the `backbone_model` level (not inside `load_model` — the hub entrypoints no longer accept it), and `load_model.repo_or_dir` should point to the **repository root** (where `hubconf.py` lives).

## `loss.py`

`MultiSegmentationLoss` provides a combined loss that can select a combination from **Cross-Entropy Loss**, **Dice Loss** and **Boundary Dice Loss**. Notably, all losses are computed as volumetric loss.

## `schedulers.py`

`WarmupOneCycleLR` realizes linear warmup with cosine annealing following.

`build_scheduler` contains various choices for schedulers, including PyTorch built-in schedulers and `WarmupOnceCycleLR`.

Notably, this file is edited basing on an original **DINOv3** **version.**

## `decoders.py`

`_get_backbone_out_indices` can select to concatenate the number of layers of outputs from a backbone，selecting from "last", "four_last", "four_even_intervals".

`ModelWithIntermediateLayers` pack up the backbone into a new class，which allows to straightly use `get_intermediate_layers` to extract the outputs from the backbone.

`ProgressiveUpHead` is a Progressively Upsampling 3D Decoder.

`UpBlock3D` is the smallest upsampling block in`ProgressiveUpHead`.

`FeatureDecoder` is a wrapper combining with a backbone and a head.

`build_segmentation_decoder`is a function which will return a `FeatureDecoder` type model.

## Datasets
### `lidc_dataset.py`

`LIDCSegDataset` is the dataset structure specified for LIDC, which returns transformed data and gt-mask as tensors.

`Transform` contains fundamental preprocessing and augmentation for LIDC, which is used in `LIDCSegDataset`.

Notably, for the gt-mask, only voxels gaining at least half votes from the experts will be marked as lesions.

### `voxel_transformer.py`

The file contains necessary transforms for the LIDC dataset, including **rotation**, **crop** and **random_center**.

### `mmwhs_dataset.py`

`MMWHS_Dataset` is the main implementation for MMWHS data, which contains augmentations and preprocessing. 

Notably, the original data also contains SDF etc. but we only adopt segmentation masks as gt for efficiency.

# Quick Start
```bash
cd PlaneCycle   # Project Root Directory

# Initialize configuration files basing on needs, there are examples for considerations.
python3 -m experiments.segmentation.seg_training --cfg "*.yaml"   
```
