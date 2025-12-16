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
    @Date   : 2023/12/5 10:19
    @Author : chairc
    @Site   : https://github.com/chairc
"""
import torch.nn as nn

from iddm.utils.logger import get_logger

logger = get_logger(name=__name__)


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
