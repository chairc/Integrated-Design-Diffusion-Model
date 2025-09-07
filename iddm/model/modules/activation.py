#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Date   : 2023/12/5 10:19
    @Author : chairc
    @Site   : https://github.com/chairc
"""
import logging
import coloredlogs
import torch.nn as nn

logger = logging.getLogger(__name__)
coloredlogs.install(level="INFO")


def get_activation_function(name="silu", inplace=False, **kwargs):
    """
    Get activation function
    :param name: Activation function name
    :param inplace: can optionally do the operation in-place
    :return Activation function
    """
    if name == "relu":
        act = nn.ReLU(inplace=inplace)
    elif name == "relu6":
        act = nn.ReLU6(inplace=inplace)
    elif name == "silu":
        act = nn.SiLU(inplace=inplace)
    elif name == "leakyrelu":
        negative_slope = kwargs.get("negative_slope", 0.2)
        act = nn.LeakyReLU(negative_slope=negative_slope, inplace=inplace)
    elif name == "gelu":
        act = nn.GELU()
    elif name == "prelu":
        act = nn.PReLU()
    elif name == "tanh":
        act = nn.Tanh()
    else:
        logger.warning(msg=f"Unsupported activation function type: {name}")
        act = nn.SiLU(inplace=inplace)
    return act
