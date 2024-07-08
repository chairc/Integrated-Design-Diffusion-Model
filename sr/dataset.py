#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Date   : 2024/1/25 15:37
    @Author : chairc
    @Site   : https://github.com/chairc
"""
import os

import torchvision

from PIL import Image
from torch.utils.data import Dataset, DataLoader, DistributedSampler

from config.setting import MEAN, STD


class SRDataset(Dataset):
    def __init__(self, image_size=64, dataset_path="", scale=4):
        self.image_size = image_size
        self.dataset_path = dataset_path
        self.image_datasets = os.listdir(self.dataset_path)
        self.scale = scale
        self.lr_transforms = torchvision.transforms.Compose([
            # Resize input size
            # torchvision.transforms.Resize(image_size)
            torchvision.transforms.Resize(size=(int(self.image_size), int(self.image_size))),
            # To Tensor Format
            torchvision.transforms.ToTensor(),
            # For standardization, the mean and standard deviation
            torchvision.transforms.Normalize(mean=MEAN, std=STD)
        ])
        self.hr_transforms = torchvision.transforms.Compose([
            # Resize input size
            # torchvision.transforms.Resize(image_size * scale)
            torchvision.transforms.Resize(size=(int(self.image_size * self.scale), int(self.image_size * self.scale))),
            # To Tensor Format
            torchvision.transforms.ToTensor(),
            # For standardization, the mean and standard deviation
            torchvision.transforms.Normalize(mean=MEAN, std=STD)
        ])

    def __getitem__(self, index):
        """
        Get sr data
        :param index: Index
        :return: lr_images, hr_images
        """
        # Resize low resolution
        image_name = self.image_datasets[index]
        image_path = os.path.join(self.dataset_path, image_name)
        image = Image.open(fp=image_path)
        image = convert_3_channels(image)
        hr_image = image.copy()
        lr_image = image.copy()
        hr_image = self.hr_transforms(hr_image)
        lr_image = self.lr_transforms(lr_image)
        return lr_image, hr_image

    def __len__(self):
        return len(self.image_datasets)


def dataset_collate(batch):
    """
    Dataset collate
    :param batch: Batch
    :return: lr_images, hr_images
    """
    lr_images = []
    hr_images = []
    for lr_image, hr_image in batch:
        lr_images.append(lr_image)
        hr_images.append(hr_image)
    return lr_images, hr_images


def convert_3_channels(image):
    if len(image.split()) != 3:
        image = image.convert("RGB")
    return image


def get_sr_dataset(image_size, dataset_path, batch_size, num_workers, distributed=False):
    dataset = SRDataset(image_size=image_size, dataset_path=dataset_path)
    if distributed:
        sampler = DistributedSampler(dataset)
        dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False,
                                num_workers=num_workers, pin_memory=True, sampler=sampler)
    else:
        dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                pin_memory=True)
    return dataloader
