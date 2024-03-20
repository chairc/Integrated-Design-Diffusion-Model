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
