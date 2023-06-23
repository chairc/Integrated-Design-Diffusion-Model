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
    U型网络
    """

    def __init__(self, in_channel=3, out_channel=3, channel=None, time_channel=256, num_classes=None, device="cpu"):
        """
        初始化UNet网络
        :param in_channel: 输入通道
        :param out_channel: 输出通道
        :param channel: 总通道列表
        :param time_channel: 时间通道
        :param num_classes: 类别数
        :param device: 使用设备
        """
        super().__init__()
        if channel is None:
            channel = [64, 128, 256, 512]
        self.device = device
        self.time_channel = time_channel
        self.inc = DoubleConv(in_channels=in_channel, out_channels=channel[0])
        self.down1 = DownBlock(in_channels=channel[0], out_channels=channel[1])
        self.sa1 = SelfAttention(channels=channel[1], size=32)
        self.down2 = DownBlock(in_channels=channel[1], out_channels=channel[2])
        self.sa2 = SelfAttention(channels=channel[2], size=16)
        self.down3 = DownBlock(in_channels=channel[2], out_channels=channel[2])
        self.sa3 = SelfAttention(channels=channel[2], size=8)

        self.bot1 = DoubleConv(in_channels=channel[2], out_channels=channel[3])
        self.bot2 = DoubleConv(in_channels=channel[3], out_channels=channel[3])
        self.bot3 = DoubleConv(in_channels=channel[3], out_channels=channel[2])

        self.up1 = UpBlock(in_channels=channel[3], out_channels=channel[1])
        self.sa4 = SelfAttention(channels=channel[1], size=16)
        self.up2 = UpBlock(in_channels=channel[2], out_channels=channel[0])
        self.sa5 = SelfAttention(channels=channel[0], size=32)
        self.up3 = UpBlock(in_channels=channel[1], out_channels=channel[0])
        self.sa6 = SelfAttention(channels=channel[0], size=64)
        self.outc = nn.Conv2d(in_channels=channel[0], out_channels=out_channel, kernel_size=1)

        if num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_channel)

    def pos_encoding(self, time, channels):
        """
        位置编码
        :param time: 时间
        :param channels: 通道
        :return: pos_enc
        """
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2, device=self.device).float() / channels))
        pos_enc_a = torch.sin(time.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(time.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, time, y=None):
        """
        前向传播
        :param x: 输入
        :param time: 时间
        :param y: 标签
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
    # 无条件
    # net = UNet(device="cpu")
    # 有条件
    net = UNet(num_classes=10, device="cpu")
    print(sum([p.numel() for p in net.parameters()]))
    x = torch.randn(3, 3, 64, 64)
    t = x.new_tensor([500] * x.shape[0]).long()
    y = x.new_tensor([1] * x.shape[0]).long()
    print(net(x, t, y).shape)
