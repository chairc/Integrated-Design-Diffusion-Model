#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Date   : 2023/6/23 22:26
    @Author : chairc
    @Site   : https://github.com/chairc
"""
import torch
import torch.nn as nn

from iddm.model.networks.base import BaseNet
from iddm.model.modules.attention import SelfAttention
from iddm.model.modules.block import DownBlock, UpBlock
from iddm.model.modules.conv import DoubleConv


class UNet(BaseNet):
    """
    UNet
    """

    def __init__(self, **kwargs):
        """
        Initialize the UNet network
        :param in_channel: Input channel
        :param out_channel: Output channel
        :param channel: The list of channel
        :param time_channel: Time channel
        :param num_classes: Number of classes
        :param image_size: Adaptive image size
        :param device: Device type
        :param act: Activation function
        """
        super(UNet, self).__init__(**kwargs)

        # channel: 3 -> 64
        # size: size
        self.inc = DoubleConv(in_channels=self.in_channel, out_channels=self.channel[1], act=self.act)

        # channel: 64 -> 128
        # size: size / 2
        self.down1 = DownBlock(in_channels=self.channel[1], out_channels=self.channel[2], act=self.act)
        # channel: 128
        # size: size / 2
        self.sa1 = SelfAttention(channels=self.channel[2], size=self.image_size_list[1], act=self.act)
        # channel: 128 -> 256
        # size: size / 4
        self.down2 = DownBlock(in_channels=self.channel[2], out_channels=self.channel[3], act=self.act)
        # channel: 256
        # size: size / 4
        self.sa2 = SelfAttention(channels=self.channel[3], size=self.image_size_list[2], act=self.act)
        # channel: 256 -> 256
        # size: size / 8
        self.down3 = DownBlock(in_channels=self.channel[3], out_channels=self.channel[3], act=self.act)
        # channel: 256
        # size: size / 8
        self.sa3 = SelfAttention(channels=self.channel[3], size=self.image_size_list[3], act=self.act)

        # channel: 256 -> 512
        # size: size / 8
        self.bot1 = DoubleConv(in_channels=self.channel[3], out_channels=self.channel[4], act=self.act)
        # channel: 512 -> 512
        # size: size / 8
        self.bot2 = DoubleConv(in_channels=self.channel[4], out_channels=self.channel[4], act=self.act)
        # channel: 512 -> 256
        # size: size / 8
        self.bot3 = DoubleConv(in_channels=self.channel[4], out_channels=self.channel[3], act=self.act)

        # channel: 512 -> 128   in_channels: up1(512) = down3(256) + bot3(256)
        # size: size / 4
        self.up1 = UpBlock(in_channels=self.channel[4], out_channels=self.channel[2], act=self.act)
        # channel: 128
        # size: size / 4
        self.sa4 = SelfAttention(channels=self.channel[2], size=self.image_size_list[2], act=self.act)
        # channel: 256 -> 64   in_channels: up2(256) = sa4(128) + sa1(128)
        # size: size / 2
        self.up2 = UpBlock(in_channels=self.channel[3], out_channels=self.channel[1], act=self.act)
        # channel: 128
        # size: size / 2
        self.sa5 = SelfAttention(channels=self.channel[1], size=self.image_size_list[1], act=self.act)
        # channel: 128 -> 64   in_channels: up3(128) = sa5(64) + inc(64)
        # size: size
        self.up3 = UpBlock(in_channels=self.channel[2], out_channels=self.channel[1], act=self.act)
        # channel: 128
        # size: size
        self.sa6 = SelfAttention(channels=self.channel[1], size=self.image_size_list[0], act=self.act)

        # channel: 64 -> 3
        # size: size
        self.outc = nn.Conv2d(in_channels=self.channel[1], out_channels=self.out_channel, kernel_size=1)

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

        bot1_out = self.bot1(x4_sa)
        bot2_out = self.bot2(bot1_out)
        bot3_out = self.bot3(bot2_out)

        up1_out = self.up1(bot3_out, x3_sa, time)
        up1_sa_out = self.sa4(up1_out)
        up2_out = self.up2(up1_sa_out, x2_sa, time)
        up2_sa_out = self.sa5(up2_out)
        up3_out = self.up3(up2_sa_out, x1, time)
        up3_sa_out = self.sa6(up3_out)
        output = self.outc(up3_sa_out)
        return output


if __name__ == "__main__":
    # Unconditional
    net = UNet(device="cpu", image_size=128)
    # Conditional
    # net = UNet(num_classes=10, device="cpu", image_size=128)
    print(sum([p.numel() for p in net.parameters()]))
    x = torch.randn(1, 3, 128, 128)
    t = x.new_tensor([500] * x.shape[0]).long()
    y = x.new_tensor([1] * x.shape[0]).long()
    print(net(x, t).shape)
    # print(net(x, t, y).shape)
