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


def get_activation_function(name="silu", inplace=False):
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
    elif name == "lrelu":
        act = nn.LeakyReLU(0.1, inplace=inplace)
    elif name == "gelu":
        act = nn.GELU()
    else:
        logger.warning(msg=f"Unsupported activation function type: {name}")
        act = nn.SiLU(inplace=inplace)
    return act
