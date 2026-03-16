import os
import random
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from torchvision.transforms import v2

from .voxel_transform import rotation, crop, random_center


class LIDCSegDataset(Dataset):
    def __init__(self, crop_size, move, data_path, train=True, upsampling_size=None,
                 original_mask=False):
        super().__init__()
        self.upsampling_size = upsampling_size
        self.data_path = data_path
        self.crop_size = crop_size
        self.move = move
        self.original_mask = original_mask

        if train:
            self.names = pd.read_csv(os.path.join(data_path, 'train_test_split.csv'))['train']. \
                dropna().map(lambda x: os.path.join(self.data_path, 'nodule', x)).values
        else:
            self.names = pd.read_csv(os.path.join(data_path, 'train_test_split.csv'))['test']. \
                dropna().map(lambda x: os.path.join(self.data_path, 'nodule', x)).values
        self.transform = Transform(crop_size, move, train, upsampling_size)

    def __getitem__(self, index):
        with np.load(self.names[index]) as npz:
            voxel = npz['voxel']

            masks = []
            counts = 0
            for i in range(1, 5):
                key = f'answer{i}'
                if key in npz:
                    masks.append(npz[key])
                    counts += 1
                else:
                    masks.append(np.zeros_like(voxel))

            masks = np.stack(masks, axis=0)

            if counts == 0:
                counts = 1
                
            gt_mask = (np.sum(masks, axis=0) >= counts / 2).astype(np.float32)

            if self.original_mask:
                return self.transform(voxel, gt_mask), gt_mask
            else:
                return self.transform(voxel, gt_mask)

    def __len__(self):
        return len(self.names)


class Transform:
    def __init__(self, size, move=None, train=True, upsampling_size=None):
        self.size = (size, size)
        self.move = move
        self.train = train
        self.upsampling_size = upsampling_size

    def __call__(self, voxel, seg):
        voxel, seg = torch.from_numpy(voxel), torch.from_numpy(seg)
        voxel = voxel.unsqueeze(1).repeat(1, 3, 1, 1)
        seg = seg.unsqueeze(1)

        voxel = voxel / 255.

        if self.upsampling_size is not None:
            voxel = F.interpolate(
                voxel,
                size=(self.upsampling_size, self.upsampling_size),
                mode='bilinear',
                align_corners=False
            ).float()
            seg = F.interpolate(
                seg.float(),
                size=(self.upsampling_size, self.upsampling_size),
                mode='nearest'
            ).long()

        shape = voxel.shape[-2:]

        if self.train:
            if self.move is not None:
                center = random_center(shape, self.move)
            else:
                center = np.array(shape) // 2
            voxel_ret = crop(voxel, center, self.size)
            seg_ret = crop(seg, center, self.size)

            rot = v2.RandomRotation(
                degrees=(-15, 15),
                interpolation=v2.InterpolationMode.NEAREST  # seg safe
            )
            voxel_ret, seg_ret = rot(voxel_ret, seg_ret)

            flip = v2.Compose([
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomVerticalFlip(p=0.5),
            ])
            voxel_ret, seg_ret = flip(voxel_ret, seg_ret)
        else:
            center = np.array(shape) // 2
            voxel_ret = crop(voxel, center, self.size)
            seg_ret = crop(seg, center, self.size)

        imagenet_normalize = v2.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        )
        voxel_ret = imagenet_normalize(voxel_ret)

        return voxel_ret, seg_ret
