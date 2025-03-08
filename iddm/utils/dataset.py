#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Date   : 2024/5/6 10:47
    @Author : chairc
    @Site   : https://github.com/chairc
"""
import torchvision

from torch.utils.data import DataLoader, DistributedSampler
from typing import Union

from iddm.config.setting import RANDOM_RESIZED_CROP_SCALE, MEAN, STD
from iddm.utils.check import check_path_is_exist


def get_dataset(image_size: Union[int, list, tuple], dataset_path=None, batch_size=2, num_workers=0, distributed=False):
    """
    Get dataset

    Automatically divide labels torchvision.datasets.ImageFolder(args.dataset_path, transform=transforms)
    If the dataset is as follow:
        dataset_path/class_1/image_1.jpg
        dataset_path/class_1/image_2.jpg
        ...
        dataset_path/class_2/image_1.jpg
        dataset_path/class_2/image_2.jpg
        ...

    'dataset_path' is the root directory of the dataset, 'class_1', 'class_2', etc. are different categories in
    the dataset, and each category contains several image files.

    Use the 'ImageFolder' class to conveniently load image datasets with this folder structure,
    and automatically assign corresponding labels to each image.

    You can specify the root directory where the dataset is located by passing the 'dataset_path' parameter,
    and perform operations such as image preprocessing and label conversion through other optional parameters.

    About Distributed Training:
    +------------------------+                     +-----------+
    |DistributedSampler      |                     |DataLoader |
    |                        |     2 indices       |           |
    |    Some strategy       +-------------------> |           |
    |                        |                     |           |
    |-------------+----------|                     |           |
                  ^                                |           |  4 data  +-------+
                  |                                |       -------------->+ train |
                1 | length                         |           |          +-------+
                  |                                |           |
    +-------------+----------+                     |           |
    |DataSet                 |                     |           |
    |        +---------+     |      3 Load         |           |
    |        |  Data   +-------------------------> |           |
    |        +---------+     |                     |           |
    |                        |                     |           |
    +------------------------+                     +-----------+

    :param image_size: Image size
    :param dataset_path: Dataset path
    :param batch_size: Batch size
    :param num_workers: Number of workers
    :param distributed: Whether to distribute training
    :return: dataloader
    """
    check_path_is_exist(path=dataset_path)
    # Data augmentation
    transforms = torchvision.transforms.Compose([
        # Resize input size, input type is (height, width)
        # torchvision.transforms.Resize(), image_size + 1/4 * image_size
        torchvision.transforms.Resize(size=set_resize_images_size(image_size=image_size, divisor=4)),
        # Random adjustment cropping
        torchvision.transforms.RandomResizedCrop(size=image_size, scale=RANDOM_RESIZED_CROP_SCALE),
        # To Tensor Format
        torchvision.transforms.ToTensor(),
        # For standardization, the mean and standard deviation
        # Refer to the initialization of ImageNet
        torchvision.transforms.Normalize(mean=MEAN, std=STD)
    ])
    # Load the folder data under the current path,
    # and automatically divide the labels according to the dataset under each file name
    dataset = torchvision.datasets.ImageFolder(root=dataset_path, transform=transforms)
    if distributed:
        sampler = DistributedSampler(dataset)
        dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                pin_memory=True, sampler=sampler)
    else:
        dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                pin_memory=True)
    return dataloader


def set_resize_images_size(image_size: Union[int, list, tuple], divisor=4):
    """
    Set resized image size
    :param image_size: Image size
    :param divisor: Divisor
    :return: image_size
    """
    if isinstance(image_size, (int, list, tuple)):
        if type(image_size) is int:
            image_size = int(image_size + image_size / divisor)
        elif type(image_size) is list:
            image_size = [int(x + x / divisor) for x in image_size]
        else:
            image_size = tuple([int(x + x / divisor) for x in image_size])
        return image_size
    else:
        raise TypeError("image_size must be int, list or tuple.")
