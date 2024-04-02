#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Date   : 2024/2/19 20:32
    @Author : chairc
    @Site   : https://github.com/chairc
"""

# Choice settings
# Support option
bool_choices = [True, False]
sample_choices = ["ddpm", "ddim", "plms"]
network_choices = ["unet", "cspdarkunet"]
optim_choices = ["adam", "adamw", "sgd"]
act_choices = ["gelu", "silu", "relu", "relu6", "lrelu"]
lr_func_choices = ["linear", "cosine", "warmup_cosine"]
image_format_choices = ["png", "jpg", "jpeg", "webp", "tif"]

# Some special parameter settings
# ****** torchvision.transforms.Compose ******
# RandomResizedCrop
RANDOM_RESIZED_CROP_SCALE = (0.8, 1.0)
# Mean in datasets
MEAN = (0.485, 0.456, 0.406)
# Std in datasets
STD = (0.229, 0.224, 0.225)
