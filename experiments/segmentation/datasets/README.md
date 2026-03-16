Note: All three files are developed basing on the efforts of **ACSConv** and **DiffAtlas**.

# `voxel_transformer.py`

The file contains necessary transforms for the LIDC dataset, including **rotation**, **crop** and **random_center**.



# `lidc_dataset.py`

`LIDCSegDataset` is the dataset structure specified for LIDC, which returns transformed data and gt-mask as tensors.

`Transform` contains fundamental preprocessing and augmentation for LIDC, which is used in `LIDCSegDataset`.

Notably, for the gt-mask, only voxels gaining at least half votes from the experts will be marked as lesions.



# `mmwhs_dataset.py`

`MMWHS_Dataset` is the main implementation for MMWHS data, which contains augmentations and preprocessing. 

Notably, the original data also contains SDF etc. but we only adopt segmentation masks as gt for efficiency.