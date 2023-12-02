#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Date   : 2023/12/2 21:28
    @Author : chairc
    @Site   : https://github.com/chairc
"""
import torch
import torch.nn as nn

from model.networks.base import BaseNet
from model.modules.module import SelfAttention, CSPDarkUpBlock, CSPDarkDownBlock, BaseConv


class CSPDarkUnet(BaseNet):
    def __init__(self, in_channel=3, out_channel=3, channel=None, time_channel=256, num_classes=None, image_size=64,
                 device="cpu", act="silu"):
        super().__init__(in_channel, out_channel, channel, time_channel, num_classes, image_size, device, act)

        # channel: 3 -> 32
        # size: size
        self.inc = BaseConv(in_channels=self.in_channel, out_channels=self.channel[0], kernel_size=1, stride=1,
                            act=self.act)

        # channel: 32 -> 64
        # size: size / 2
        self.down1 = CSPDarkDownBlock(in_channels=self.channel[0], out_channels=self.channel[1], n=1, act=self.act)
        # channel: 64
        # size: size / 2
        self.sa1 = SelfAttention(channels=self.channel[1], size=int(self.image_size / 2), act=self.act)
        # channel: 3 -> 64
        # size: size / 4
        self.down2 = CSPDarkDownBlock(in_channels=self.channel[1], out_channels=self.channel[2], n=3, act=self.act)
        # channel: 128
        # size: size / 4
        self.sa2 = SelfAttention(channels=self.channel[2], size=int(self.image_size / 4), act=self.act)
        # channel: 3 -> 64
        # size: size / 8
        self.down3 = CSPDarkDownBlock(in_channels=self.channel[2], out_channels=self.channel[3], n=3, act=self.act)
        # channel: 256
        # size: size / 8
        self.sa3 = SelfAttention(channels=self.channel[3], size=int(self.image_size / 8), act=self.act)
        # channel: 3 -> 64
        # size: size / 16
        self.down4 = CSPDarkDownBlock(in_channels=self.channel[3], out_channels=self.channel[4], n=1, act=self.act)
        # channel: 512
        # size: size / 16
        self.sa4 = SelfAttention(channels=self.channel[4], size=int(self.image_size / 16), act=self.act)

        # channel: 512 -> 256
        # size: size / 8
        self.up1 = CSPDarkUpBlock(in_channels=self.channel[4], out_channels=self.channel[3], n=3, act=self.act)
        # channel: 256
        # size: size / 8
        self.sa5 = SelfAttention(channels=self.channel[3], size=int(self.image_size / 8), act=self.act)
        # channel: 256 -> 128
        # size: size / 4
        self.up2 = CSPDarkUpBlock(in_channels=self.channel[3], out_channels=self.channel[2], n=3, act=self.act)
        # channel: 128
        # size: size / 4
        self.sa6 = SelfAttention(channels=self.channel[2], size=int(self.image_size / 4), act=self.act)
        # channel: 128 -> 64
        # size: size / 2
        self.up3 = CSPDarkUpBlock(in_channels=self.channel[2], out_channels=self.channel[1], n=3, act=self.act)
        # channel: 64
        # size: size / 2
        self.sa7 = SelfAttention(channels=self.channel[1], size=int(self.image_size / 2), act=self.act)
        # channel: 64 -> 32
        # size: size
        self.up4 = CSPDarkUpBlock(in_channels=self.channel[1], out_channels=self.channel[0], n=3, act=self.act)
        # channel: 32
        # size: size
        self.sa8 = SelfAttention(channels=self.channel[0], size=int(self.image_size), act=self.act)

        # channel: 32 -> 3
        # size: size
        self.outc = nn.Conv2d(in_channels=self.channel[0], out_channels=self.out_channel, kernel_size=1)

    def forward(self, x, time, y=None):
        """
        Forward
        :param x: Input
        :param time: Time
        :param y: Input label
        :return: output
        """
        time = time.unsqueeze(-1).type(torch.float)
        time = self.pos_encoding(time, self.time_channel)

        if y is not None:
            time += self.label_emb(y)

        x1 = self.inc(x)
        x2 = self.down1(x1, time)
        x2_sa = self.sa1(x2)
        x3 = self.down2(x2_sa, time)
        x3_sa = self.sa2(x3)
        x4 = self.down3(x3_sa, time)
        x4_sa = self.sa3(x4)
        x5 = self.down4(x4_sa, time)
        x5_sa = self.sa4(x5)

        up1_out = self.up1(x5_sa, x4_sa, time)
        up1_sa_out = self.sa5(up1_out)
        up2_out = self.up2(up1_sa_out, x3_sa, time)
        up2_sa_out = self.sa6(up2_out)
        up3_out = self.up3(up2_sa_out, x2_sa, time)
        up3_sa_out = self.sa7(up3_out)
        up4_out = self.up4(up3_sa_out, x1, time)
        up4_sa_out = self.sa8(up4_out)
        output = self.outc(up4_sa_out)

        return output
