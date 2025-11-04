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
    @Date   : 2023/12/5 10:22
    @Author : chairc
    @Site   : https://github.com/chairc
"""
import logging
import coloredlogs

import torch
import torch.nn as nn
import torch.nn.functional as F

from iddm.model.modules.activation import get_activation_function

logger = logging.getLogger(__name__)
coloredlogs.install(level="INFO")


class DoubleConv(nn.Module):
    """
    Double convolution
    """

    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False, act="silu"):
        """
        Initialize the double convolution block
        :param in_channels: Input channels
        :param out_channels: Output channels
        :param mid_channels: Middle channels
        :param residual: Whether residual
        :param act: Activation function
        """
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.act = act
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=1, num_channels=mid_channels),
            get_activation_function(name=self.act),
            nn.Conv2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=1, num_channels=out_channels),
        )

    def forward(self, x):
        """
        DoubleConv forward
        :param x: Input
        :return: Residual or non-residual results
        """
        if self.residual:
            out = x + self.double_conv(x)
            if self.act == "relu":
                return F.relu(out)
            elif self.act == "relu6":
                return F.relu6(out)
            elif self.act == "silu":
                return F.silu(out)
            elif self.act == "lrelu":
                return F.leaky_relu(out)
            elif self.act == "gelu":
                return F.gelu(out)
            else:
                logger.warning(msg=f"Unsupported activation function type: {self.act}")
                return F.silu(out)
        else:
            return self.double_conv(x)


class BaseConv(nn.Module):
    """
    Base convolution
    Conv2d -> BatchNorm -> Activation function block
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, groups=1, bias=False, act="silu"):
        """
        Initialize the Base convolution
        :param in_channels: Input channels
        :param out_channels: Output channels
        :param kernel_size: Kernel size
        :param stride: Stride
        :param groups: Groups
        :param bias: Bias
        :param act: Activation function
        """
        super().__init__()
        # Same padding
        pad = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              stride=stride, padding=pad, groups=groups, bias=bias)
        self.gn = nn.GroupNorm(num_groups=1, num_channels=out_channels)
        self.act = get_activation_function(name=act, inplace=True)

    def forward(self, x):
        return self.act(self.gn(self.conv(x)))


class VAEConv2d(nn.Module):
    """
    VAE Convolutional
    """

    def __init__(
            self, in_channels: int,
            out_channels: int,
            kernel_size: int = 3,
            stride: int = 1,
            padding: int = 1,
            downsample: bool = False,
            act: str = "silu",
    ):
        super().__init__()
        self.stride = 2 if downsample else stride
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=self.stride,
            padding=padding
        )
        self.norm = nn.GroupNorm(num_groups=32, num_channels=out_channels)
        self.act = get_activation_function(name=act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x
