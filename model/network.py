#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Date   : 2023/6/23 22:26
    @Author : chairc
    @Site   : https://github.com/chairc
"""
import torch
import torch.nn as nn

from model.modules import UpBlock, DownBlock, DoubleConv, SelfAttention


class UNet(nn.Module):
    """
    UNet
    """

    def __init__(self, in_channel=3, out_channel=3, channel=None, time_channel=256, num_classes=None, image_size=64,
                 device="cpu", act="silu"):
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
        super().__init__()
        if channel is None:
            channel = [64, 128, 256, 512]
        self.device = device
        self.time_channel = time_channel
        self.image_size = image_size
        # channel: 3 -> 64
        # size: size
        self.inc = DoubleConv(in_channels=in_channel, out_channels=channel[0], act=act)

        # channel: 64 -> 128
        # size: size / 2
        self.down1 = DownBlock(in_channels=channel[0], out_channels=channel[1], act=act)
        # channel: 128
        # size: size / 2
        self.sa1 = SelfAttention(channels=channel[1], size=int(self.image_size / 2), act=act)
        # channel: 128 -> 256
        # size: size / 4
        self.down2 = DownBlock(in_channels=channel[1], out_channels=channel[2], act=act)
        # channel: 256
        # size: size / 4
        self.sa2 = SelfAttention(channels=channel[2], size=int(self.image_size / 4), act=act)
        # channel: 256 -> 256
        # size: size / 8
        self.down3 = DownBlock(in_channels=channel[2], out_channels=channel[2], act=act)
        # channel: 256
        # size: size / 8
        self.sa3 = SelfAttention(channels=channel[2], size=int(self.image_size / 8), act=act)

        # channel: 256 -> 512
        # size: size / 8
        self.bot1 = DoubleConv(in_channels=channel[2], out_channels=channel[3], act=act)
        # channel: 512 -> 512
        # size: size / 8
        self.bot2 = DoubleConv(in_channels=channel[3], out_channels=channel[3], act=act)
        # channel: 512 -> 256
        # size: size / 8
        self.bot3 = DoubleConv(in_channels=channel[3], out_channels=channel[2], act=act)

        # channel: 512 -> 128   in_channels: up1(512) = down3(256) + bot3(256)
        # size: size / 4
        self.up1 = UpBlock(in_channels=channel[3], out_channels=channel[1], act=act)
        # channel: 128
        # size: size / 4
        self.sa4 = SelfAttention(channels=channel[1], size=int(self.image_size / 4), act=act)
        # channel: 256 -> 64   in_channels: up2(256) = sa4(128) + sa1(128)
        # size: size / 4
        self.up2 = UpBlock(in_channels=channel[2], out_channels=channel[0], act=act)
        # channel: 128
        # size: size / 2
        self.sa5 = SelfAttention(channels=channel[0], size=int(self.image_size / 2), act=act)
        # channel: 128 -> 64   in_channels: up3(128) = sa5(64) + inc(64)
        # size: size / 4
        self.up3 = UpBlock(in_channels=channel[1], out_channels=channel[0], act=act)
        # channel: 128
        # size: size
        self.sa6 = SelfAttention(channels=channel[0], size=int(self.image_size), act=act)

        # channel: 64 -> 3
        # size: size
        self.outc = nn.Conv2d(in_channels=channel[0], out_channels=out_channel, kernel_size=1)

        if num_classes is not None:
            self.label_emb = nn.Embedding(num_embeddings=num_classes, embedding_dim=time_channel)

    def pos_encoding(self, time, channels):
        """
        Position encoding
        :param time: Time
        :param channels: Channels
        :return: pos_enc
        """
        inv_freq = 1.0 / (10000 ** (torch.arange(start=0, end=channels, step=2, device=self.device).float() / channels))
        inv_freq_value = time.repeat(1, channels // 2) * inv_freq
        pos_enc_a = torch.sin(input=inv_freq_value)
        pos_enc_b = torch.cos(input=inv_freq_value)
        pos_enc = torch.cat(tensors=[pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

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
        x2 = self.sa1(x2)
        x3 = self.down2(x2, time)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, time)
        x4 = self.sa3(x4)

        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        output = self.up1(x4, x3, time)
        output = self.sa4(output)
        output = self.up2(output, x2, time)
        output = self.sa5(output)
        output = self.up3(output, x1, time)
        output = self.sa6(output)
        output = self.outc(output)
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
