#!/usr/bin/env python
# -*- coding:utf-8 -*-

# Copyright 2025 IDDM Authors

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
    @Date   : 2023/7/15 23:50
    @Author : chairc
    @Site   : https://github.com/chairc
"""
import math


def set_cosine_lr(optimizer, current_epoch, max_epoch, lr_min=0, lr_max=0.1, warmup=True, num_warmup=5):
    """
    Set the optimizer learning rate
    :param optimizer: Optimizer
    :param current_epoch: Current epoch
    :param max_epoch: Max epoch
    :param lr_min: Min learning rate
    :param lr_max: Max learning rate
    :param warmup: Whether to warmup
    :param num_warmup: Number of warmup
    :return: lr

    """
    warmup_epoch = num_warmup if warmup else 0
    if current_epoch < warmup_epoch:
        lr = lr_max * current_epoch / warmup_epoch
    elif current_epoch < max_epoch:
        lr = lr_min + (lr_max - lr_min) * (
                1 + math.cos(math.pi * (current_epoch - warmup_epoch) / (max_epoch - warmup_epoch))) / 2
    else:
        lr = lr_min + (lr_max - lr_min) * (
                1 + math.cos(math.pi * (current_epoch - max_epoch) / max_epoch)) / 2
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr
