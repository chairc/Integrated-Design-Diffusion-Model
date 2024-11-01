#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Date   : 2024/7/9 00:01
    @Author : chairc
    @Site   : https://github.com/chairc
"""
from config.choices import image_type_choices

# Train
MASTER_ADDR = "localhost"
MASTER_PORT = "12345"
EMA_BETA = 0.995

# Some special parameter settings
# model.networks
# Image type setting (1 channel or 3 channels), you can set "RGB" or "GRAY" in the project
IMAGE_TYPE = "RGB"
IMAGE_CHANNEL = image_type_choices[IMAGE_TYPE]
TIME_CHANNEL = 256
CHANNEL_LIST = [32, 64, 128, 256, 512, 1024]

DEFAULT_IMAGE_SIZE = [64, 64]

# model.networks.sr
SR_IMAGE_TYPE = "RGB"
SR_IMAGE_CHANNEL = image_type_choices[SR_IMAGE_TYPE]

# Data processing
# ****** torchvision.transforms.Compose ******
# RandomResizedCrop
RANDOM_RESIZED_CROP_SCALE = (0.8, 1.0)
# Mean in datasets
MEAN = (0.485, 0.456, 0.406)
# Std in datasets
STD = (0.229, 0.224, 0.225)
