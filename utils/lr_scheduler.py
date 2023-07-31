#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Date   : 2023/7/15 23:50
    @Author : chairc
    @Site   : https://github.com/chairc
"""
import math


def set_cosine_lr(optimizer, current_epoch, max_epoch, lr_min=0, lr_max=0.1, warmup=True, num_warmup=5):
    """
    设置优化器学习率
    :param optimizer: 优化器
    :param current_epoch: 当前迭代次数
    :param max_epoch: 最大迭代次数
    :param lr_min: 最小学习率
    :param lr_max: 最大学习率
    :param warmup: 预热
    :param num_warmup: 预热个数
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
