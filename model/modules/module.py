#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Date   : 2023/6/15 17:12
    @Author : chairc
    @Site   : https://github.com/chairc
"""
import logging
import coloredlogs

import torch
import torch.nn as nn

from model.modules.conv import BaseConv
from model.modules.activation import get_activation_function

logger = logging.getLogger(__name__)
coloredlogs.install(level="INFO")


class Bottleneck(nn.Module):
    """
    Standard bottleneck
    """

    def __init__(self, in_channels, out_channels, shortcut=True, expansion=0.5, act="silu"):
        """
        Initialize the Bottleneck
        :param in_channels: Input channels
        :param out_channels: Output channels
        :param shortcut: Shortcut, such as x + y
        :param expansion: Factor
        :param act: Activation function
        """
        super().__init__()
        mid_channels = int(out_channels * expansion)
        self.conv1 = BaseConv(in_channels=in_channels, out_channels=mid_channels, kernel_size=1, stride=1, act=act)
        self.conv2 = BaseConv(in_channels=mid_channels, out_channels=out_channels, kernel_size=3, stride=1, act=act)
        # Use shortcut
        self.use_add = shortcut and in_channels == out_channels

    def forward(self, x):
        y = self.conv2(self.conv1(x))

        if self.use_add:
            y = y + x
        return y


class SPPFBottleneck(nn.Module):
    """
    SPPF Bottleneck
    https://github.com/ultralytics/yolov5/blob/master/models/common.py
    https://github.com/ultralytics/yolov5/blob/3eefab1bb109214a614485b6c5f80f22c122f2b2/models/common.py#L182
    """

    # 'kernel_sizes = 5' is the same as 'SPP(kernel_sizes=(5, 9, 13))'
    def __init__(self, in_channels, out_channels, kernel_size=5, act="silu"):
        """
        Initialize the SPPFBottleneck
        :param in_channels: Input channels
        :param out_channels: Output channels
        :param kernel_size: Kernel size
        :param act: Activation function
        """
        super().__init__()
        mid_channels = in_channels // 2
        self.conv1 = BaseConv(in_channels=in_channels, out_channels=mid_channels, kernel_size=1, stride=1, act=act)
        self.conv2 = BaseConv(in_channels=mid_channels * 4, out_channels=out_channels, kernel_size=1, stride=1, act=act)
        self.m = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

    def forward(self, x):
        x = self.conv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.conv2(torch.cat([x, y1, y2, self.m(y2)], dim=1))


class CSPLayer(nn.Module):
    """
    CSP Bottleneck with 3 convolutions
    """

    def __init__(self, in_channels, out_channels, n=1, shortcut=True, expansion=0.5, act="silu"):
        """
        Initialize the CSPLayer
        :param in_channels: Input channels
        :param out_channels: Output channels
        :param n: Number of Bottlenecks
        :param shortcut: Shortcut, such as x + y
        :param expansion: Factor
        :param act: Activation function
        """
        super().__init__()
        mid_channels = int(out_channels * expansion)  # hidden channels
        self.conv1 = BaseConv(in_channels=in_channels, out_channels=mid_channels, kernel_size=1, stride=1, act=act)
        self.conv2 = BaseConv(in_channels=in_channels, out_channels=mid_channels, kernel_size=1, stride=1, act=act)
        self.conv3 = BaseConv(in_channels=2 * mid_channels, out_channels=out_channels, kernel_size=1, stride=1, act=act)
        module_list = [
            Bottleneck(in_channels=mid_channels, out_channels=mid_channels, shortcut=shortcut, expansion=1.0,
                       act=act)
            for _ in range(n)
        ]
        self.m = nn.Sequential(*module_list)

    def forward(self, x):
        x_1 = self.conv1(x)
        x_2 = self.conv2(x)
        x_1 = self.m(x_1)
        x = torch.cat([x_1, x_2], dim=1)
        return self.conv3(x)


class DenseModule(nn.Module):
    def __init__(self, in_channels, out_channels, act="silu"):
        """
        Initialize the DenseModule
        :param in_channels: Input channels
        :param out_channels: Output channels
        :param act: Activation function
        """
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=False)
        self.act = get_activation_function(name=act)

    def forward(self, x):
        y = self.conv(x)
        y = self.act(y)
        y = torch.cat([x, y], dim=1)
        return y
