"""3D input transforms for the MedMNIST experiments."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import v2


class _Aug3DBase(nn.Module):
    """Shared spatial augmentation pipeline: optional resize to
    target_resolution, then (train only) random crop / flips / 90-degree
    rotations. Subclasses decide the final channel/normalization treatment.
    """

    def __init__(self, mode="val", resolution=64, target_resolution=64):
        super().__init__()
        self.mode = "train" if "train" in mode else "val"

        need_resize = target_resolution != resolution
        pad_size = max(1, 4 * (target_resolution // resolution)) if need_resize else 4

        def resize():
            return v2.Resize(
                size=target_resolution, interpolation=v2.InterpolationMode.BILINEAR
            )

        core_spatial_aug = [
            v2.RandomCrop(
                size=target_resolution, padding=pad_size, padding_mode="reflect"
            ),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
            v2.RandomChoice(
                [
                    v2.RandomRotation(degrees=(90, 90)),
                    v2.RandomRotation(degrees=(180, 180)),
                    v2.RandomRotation(degrees=(270, 270)),
                    v2.Identity(),
                ]
            ),
        ]

        if need_resize:
            self.train_aug = v2.Compose([resize(), *core_spatial_aug])
            self.val_test_aug = resize()
        else:
            self.train_aug = v2.Compose(core_spatial_aug)
            self.val_test_aug = v2.Identity()

    def _prep(self, x):
        """To a float tensor in [0, 1], shaped (C, D, H, W), augmented per mode."""
        if not isinstance(x, torch.Tensor):
            x = np.asarray(x) if isinstance(x, Image.Image) else x
            x = torch.from_numpy(x).float()
        if x.max() > 100:
            x = x / 255.0
        if x.ndim == 3:
            x = x.unsqueeze(0)
        elif x.ndim == 2:
            x = x.unsqueeze(0).unsqueeze(0)
        return self.train_aug(x) if self.mode == "train" else self.val_test_aug(x)


class Lifted2DTransform(_Aug3DBase):
    """Input for lifted 2D foundation models (DINOv3 ViT/ConvNeXt, any
    block_type): 3-channel (grayscale repeated), ImageNet-normalized —
    the input format the pretrained 2D weights expect."""

    def __init__(self, mode="val", resolution=64, target_resolution=64):
        super().__init__(mode, resolution, target_resolution)
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)
        )

    def forward(self, x):
        x = self._prep(x)
        if x.shape[0] == 1:
            x = x.expand(3, -1, -1, -1)
        return (x - self.mean) / self.std


class Native3DTransform(_Aug3DBase):
    """Input for natively-3D models (SPECTRE, CT-FM): single channel,
    values in [0, 1] — no RGB expansion, no ImageNet normalization."""

    def forward(self, x):
        return self._prep(x)


def make_tri_slice(x):
    """Replace 3 identical grayscale channels with (prev, curr, next) slices
    — the 2.5D "TriSlice" input encoding.

    Args:
        x: (B, 3, D, H, W) — channels are replicated grayscale
    Returns:
        (B, 3, D, H, W) — channels are slice d-1, d, d+1 (replicate-padded)
    """
    gray = x[:, 0:1, :, :, :]  # (B, 1, D, H, W)
    # replicate-pad D by 1 on each side → (B, 1, D+2, H, W)
    gray_pad = F.pad(gray, (0, 0, 0, 0, 1, 1), mode="replicate")
    D = x.shape[2]
    prev = gray_pad[:, :, 0:D, :, :]
    curr = gray_pad[:, :, 1 : D + 1, :, :]
    next_ = gray_pad[:, :, 2 : D + 2, :, :]
    return torch.cat([prev, curr, next_], dim=1)  # (B, 3, D, H, W)


def upsample_hw(x, scale_factor=2, mode="trilinear"):
    """
    Upsample only H and W dimensions, keep D unchanged.

    "trilinear" is required by F.interpolate for 5D input, but with the D
    scale factor fixed at 1 it is mathematically identical to a per-slice
    bilinear resize — no mixing across slices ever happens.

    Args:
        x: (B, C, D, H, W)
        scale_factor: int or float
        mode: "trilinear" or "nearest"

    Returns:
        x: (B, C, D, H*scale, W*scale)
    """
    x = F.interpolate(
        x,
        scale_factor=(1, scale_factor, scale_factor),
        mode=mode,
        align_corners=False if mode in ["trilinear", "bilinear"] else None,
    )
    return x
