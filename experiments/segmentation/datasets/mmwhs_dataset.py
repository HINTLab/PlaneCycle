import os
import numpy as np
import torch
from torch.utils.data.dataset import Dataset

import glob
import cv2
import torchio as tio
from torch.utils.data import DataLoader
from torchvision.transforms import v2

PREPROCESSING_TRANSORMS_CT = tio.Compose([
    tio.Clamp(out_min=-250, out_max=800),
    tio.RescaleIntensity(in_min_max=(-250, 800),
                         out_min_max=(0.0, 1.0)),
    tio.CropOrPad(target_shape=(64, 64, 64))
])
PREPROCESSING_TRANSORMS_MRI = tio.Compose([
    tio.Clamp(out_min=0, out_max=1000),
    tio.RescaleIntensity(in_min_max=(0, 1000),
                         out_min_max=(0.0, 1.0)),
    tio.CropOrPad(target_shape=(64, 64, 64))
])

PREPROCESSING_MASK_TRANSORMS = tio.Compose([
    tio.CropOrPad(target_shape=(64, 64, 64))
])


class MMWHS_Dataset(Dataset):
    def __init__(self, root_dir='', data_type='', mode=''):
        self.root_dir = root_dir
        self.data_type = data_type
        self.mode = mode
        self.file_names = self.get_file_names()
        self.preprocessing_img_ct = PREPROCESSING_TRANSORMS_CT
        self.preprocessing_img_mri = PREPROCESSING_TRANSORMS_MRI
        self.preprocessing_mask = PREPROCESSING_MASK_TRANSORMS

    def train_transform(self, image, label, sdf, p=0.5):

        TRAIN_TRANSFORMS = tio.Compose([
            tio.RandomFlip(
                axes=(0, 1, 2),
                flip_probability=p,
            ),

            tio.RandomAffine(
                scales=(0.9, 1.1),
                degrees=10,
                translation=5,
                image_interpolation='linear',
                p=p,
            ),

            tio.RandomElasticDeformation(
                num_control_points=7,
                max_displacement=5,
                image_interpolation='linear',
                p=p * 0.5,
            ),

            tio.RandomGamma(
                log_gamma=(-0.3, 0.3),
                p=p * 0.5,
            ),
            tio.RandomNoise(
                mean=0,
                std=0.01,
                p=p * 0.5,
            ),
            tio.RandomBlur(
                std=(0, 1.0),
                p=p * 0.3,
            ),
        ])

        subject = tio.Subject(
            image=image,
            label=label,
            sdf=sdf,
        )

        subject = TRAIN_TRANSFORMS(subject)

        return subject.image.tensor, subject.label.tensor, subject.sdf.tensor

    def get_file_names(self):
        all_img_names = glob.glob(os.path.join(self.root_dir, './*image.nii.gz'), recursive=True)
        return all_img_names

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, index):
        img_path = self.file_names[index]
        mask_path = img_path.replace("image.nii.gz", "label.nii.gz")
        sdf_path = img_path.replace("image.nii.gz", "sdf.nii.gz")

        img = tio.ScalarImage(img_path)
        mask = tio.LabelMap(mask_path)
        sdf = tio.ScalarImage(sdf_path)
        name = img_path.split('/')[-1]
        name = name.split('.nii')[0]

        if self.data_type.upper() == 'CT':
            img = self.preprocessing_img_ct(img)
        elif self.data_type.upper() == 'MRI':
            img = self.preprocessing_img_mri(img)
        else:
            raise ValueError("Wrong Data Type!")
        mask = self.preprocessing_mask(mask)

        p = np.random.choice([0, 1])
        if self.mode == 'train':
            img, mask, sdf = self.train_transform(img, mask, sdf, p)

        mask = mask.data
        img = img.data
        sdf = sdf.data

        sdf = sdf.squeeze(3)
        sdf = sdf.permute(1, 0, 2, 3)

        label_0 = (mask == 0).float()
        label_1 = (mask == 1).float()
        label_2 = (mask == 2).float()
        label_3 = (mask == 3).float()
        label_4 = (mask == 4).float()
        label_5 = (mask == 5).float()
        label = torch.cat((label_0, label_1, label_2, label_3, label_4, label_5), dim=0)

        imagenet_normalize = v2.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        )
        img = imagenet_normalize(img.permute(1, 0, 2, 3).repeat(1, 3, 1, 1))

        label = label.permute(1, 0, 2, 3)

        return img, label
