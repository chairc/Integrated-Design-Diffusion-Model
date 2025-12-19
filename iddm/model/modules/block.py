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
    @Date   : 2023/12/5 10:21
    @Author : chairc
    @Site   : https://github.com/chairc
"""
import torch
import torch.nn as nn

from iddm.config.setting import EMB_CHANNEL
from iddm.model.modules.conv import BaseConv, DoubleConv, VAEConv2d
from iddm.model.modules.module import CSPLayer, DenseModule


class DownBlock(nn.Module):
    """
    Downsample block
    """

    def __init__(self, in_channels, out_channels, emb_channels=EMB_CHANNEL, act="silu"):
        """
        Initialize the downsample block
        :param in_channels: Input channels
        :param out_channels: Output channels
        :param emb_channels: Embed channels
        :param act: Activation function
        """
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels=in_channels, out_channels=in_channels, residual=True, act=act),
            DoubleConv(in_channels=in_channels, out_channels=out_channels, act=act),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(in_features=emb_channels, out_features=out_channels),
        )

    def forward(self, x, time):
        """
        DownBlock forward
        :param x: Input
        :param time: Time
        :return: x + emb
        """
        x = self.maxpool_conv(x)
        emb = self.emb_layer(time)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class UpBlock(nn.Module):
    """
    Upsample Block
    """

    def __init__(self, in_channels, out_channels, emb_channels=EMB_CHANNEL, act="silu"):
        """
        Initialize the upsample block
        :param in_channels: Input channels
        :param out_channels: Output channels
        :param emb_channels: Embed channels
        :param act: Activation function
        """
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels=in_channels, out_channels=in_channels, residual=True, act=act),
            DoubleConv(in_channels=in_channels, out_channels=out_channels, mid_channels=in_channels // 2, act=act),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(in_features=emb_channels, out_features=out_channels),
        )

    def forward(self, x, skip_x, time):
        """
        UpBlock forward
        :param x: Input
        :param skip_x: Merged input
        :param time: Time
        :return: x + emb
        """
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(time)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class UpBlockV2(UpBlock):
    """
    Upsample Block v2
    """

    def __init__(self, in_channels, out_channels, emb_channels=EMB_CHANNEL, act="silu"):
        """
        Initialize the upsample block v2
        :param in_channels: Input channels
        :param out_channels: Output channels
        :param emb_channels: Embed channels
        :param act: Activation function
        """
        super().__init__(in_channels, out_channels, emb_channels, act)
        self.mid_channels = int(in_channels / 2)
        self.up = nn.ConvTranspose2d(in_channels=self.mid_channels, out_channels=self.mid_channels, kernel_size=2,
                                     stride=2)

    def forward(self, x, skip_x, time):
        """
        UpBlock forward
        :param x: Input
        :param skip_x: Merged input
        :param time: Time
        :return: x + emb
        """
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(time)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class CSPDarkDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, emb_channels=EMB_CHANNEL, n=1, act="silu"):
        super().__init__()
        self.conv_csp = nn.Sequential(
            BaseConv(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2, act=act),
            CSPLayer(in_channels=out_channels, out_channels=out_channels, n=n, act=act)
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(in_features=emb_channels, out_features=out_channels),
        )

    def forward(self, x, time):
        x = self.conv_csp(x)
        emb = self.emb_layer(time)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class CSPDarkUpBlock(nn.Module):

    def __init__(self, in_channels, out_channels, emb_channels=EMB_CHANNEL, n=1, act="silu"):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = BaseConv(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, act=act)
        self.csp = CSPLayer(in_channels=in_channels, out_channels=out_channels, n=n, shortcut=False, act=act)

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(in_features=emb_channels, out_features=out_channels),
        )

    def forward(self, x, skip_x, time):
        x = self.conv(x)
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(time)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class ResidualDenseBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n=1, act="silu"):
        super().__init__()
        mid_channels = in_channels
        module_list = []
        for _ in range(n):
            module_list.append(DenseModule(in_channels=mid_channels, out_channels=out_channels, act=act))
            mid_channels += out_channels
        self.m = nn.Sequential(*module_list)
        self.conv = nn.Conv2d(in_channels=mid_channels, out_channels=in_channels, kernel_size=1, padding=0, bias=False)

    def forward(self, x):
        y = self.m(x)
        y = self.conv(y)
        return x + y


class VAEResidualBlock(nn.Module):
    """
    VAE Residual Block: Enhanced feature propagation capabilities
    """

    def __init__(self, in_channels: int, out_channels: int, act: str = "silu"):
        super().__init__()
        self.block1 = VAEConv2d(in_channels=in_channels, out_channels=out_channels, downsample=False, act=act)
        self.block2 = VAEConv2d(in_channels=out_channels, out_channels=out_channels, downsample=False, act=act)
        self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                                       padding=0) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual_conv(x)
        x = self.block1(x)
        x = self.block2(x)
        return x + residual


class VAEUpBlock(nn.Module):
    """
    Upsampling block: Used to scale up the feature size of the decoder
    """

    def __init__(self, in_channels: int, out_channels: int, act: str = "silu"):
        super().__init__()
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=2)
        self.conv1 = VAEConv2d(in_channels, out_channels * 4, downsample=False, act=act)
        self.conv2 = VAEConv2d(out_channels, out_channels, downsample=False, act=act)
        # Gaussian smoothing kernels to relieve boundary aliasing
        self.smooth = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, padding_mode="reflect")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.pixel_shuffle(x)
        x = self.conv2(x)
        x = self.smooth(x)
        return x
