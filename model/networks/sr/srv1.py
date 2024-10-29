#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Date   : 2024/1/23 21:49
    @Author : chairc
    @Site   : https://github.com/chairc
"""
import torch
import torch.nn as nn

from config.setting import SR_IMAGE_CHANNEL
from model.modules.block import ResidualDenseBlock


class SRv1(nn.Module):
    """
    Super resolution v1
    Residual Dense Network
    """

    def __init__(self, in_channel=SR_IMAGE_CHANNEL, out_channel=SR_IMAGE_CHANNEL, channel=None, n=6, scale=4,
                 act="silu"):
        """
        The implement of RDN
        Paper: Residual Dense Network for Image Super-Resolution
        URL: https://arxiv.org/abs/1802.08797
        :param in_channel: Input channel
        :param out_channel: Output channel
        :param channel: Middle channel
        :param n: Number of Residual Blocks
        :param scale: Scale of Residual Blocks
        :param act: Activation function
        """
        super().__init__()
        # Default
        if channel is None:
            channel = [64]
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.channel = channel
        self.n = n
        self.scale = scale
        self.act = act

        # Initial feature extraction
        self.conv1 = nn.Conv2d(in_channels=self.in_channel, out_channels=self.channel[0], kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=self.channel[0], out_channels=self.channel[0], kernel_size=1, padding=0)

        # Back-projection stages
        self.stage1 = ResidualDenseBlock(in_channels=self.channel[0], out_channels=int(self.channel[0] / 2), n=n,
                                         act=self.act)
        self.stage2 = ResidualDenseBlock(in_channels=self.channel[0], out_channels=int(self.channel[0] / 2), n=n,
                                         act=self.act)
        self.stage3 = ResidualDenseBlock(in_channels=self.channel[0], out_channels=int(self.channel[0] / 2), n=n,
                                         act=self.act)

        # Global feature fusion
        self.gff1 = nn.Conv2d(self.channel[0] * 3, out_channels=self.channel[0], kernel_size=1, padding=0)
        self.gff2 = nn.Conv2d(self.channel[0], out_channels=self.channel[0], kernel_size=3, padding=1)

        # Pixel upsample
        self.up_conv = nn.Conv2d(in_channels=self.channel[0], out_channels=self.channel[0] * self.scale * self.scale,
                                 kernel_size=3, padding=1)
        self.upsample = nn.PixelShuffle(upscale_factor=self.scale)

        # Output reconstruction images
        self.conv3 = nn.Conv2d(in_channels=self.channel[0], out_channels=self.out_channel, kernel_size=3, padding=1)

    def forward(self, x):
        # Initial feature extraction
        out1 = self.conv1(x)
        out2 = self.conv2(out1)

        # Back-projection stages
        s1 = self.stage1(out2)
        s2 = self.stage2(s1)
        s3 = self.stage3(s2)
        s = torch.cat([s1, s2, s3], dim=1)

        # Global feature fusion
        gf1 = self.gff1(s)
        gf2 = self.gff2(gf1)

        # Pixel upsample
        gf = out1 + gf2
        up_out1 = self.up_conv(gf)
        up_out2 = self.upsample(up_out1)

        output = self.conv3(up_out2)

        return output


if __name__ == "__main__":
    srv1 = SRv1()
    x = torch.randn(1, 3, 64, 64)
    print(srv1(x))
    print(srv1(x).shape)
