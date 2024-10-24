#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Date   : 2023/6/23 22:26
    @Author : chairc
    @Site   : https://github.com/chairc
"""
import torch

from model.networks.unet import UNet
from model.modules.block import UpBlockV2


class UNetV2(UNet):
    """
    UNetV2
    Replace nn.Upsample with nn.ConvTranspose2d
    """

    def __init__(self, **kwargs):
        """
        Initialize the UNetV2 network
        :param in_channel: Input channel
        :param out_channel: Output channel
        :param channel: The list of channel
        :param time_channel: Time channel
        :param num_classes: Number of classes
        :param image_size: Adaptive image size
        :param device: Device type
        :param act: Activation function
        """
        super(UNetV2, self).__init__(**kwargs)

        # channel: 512 -> 128   in_channels: up1(512) = down3(256) + bot3(256)
        # size: size / 4
        self.up1 = UpBlockV2(in_channels=self.channel[4], out_channels=self.channel[2], act=self.act)
        # channel: 256 -> 64   in_channels: up2(256) = sa4(128) + sa1(128)
        # size: size / 2
        self.up2 = UpBlockV2(in_channels=self.channel[3], out_channels=self.channel[1], act=self.act)
        # channel: 128 -> 64   in_channels: up3(128) = sa5(64) + inc(64)
        # size: size
        self.up3 = UpBlockV2(in_channels=self.channel[2], out_channels=self.channel[1], act=self.act)


if __name__ == "__main__":
    # Unconditional
    net = UNetV2(device="cpu", image_size=128)
    # Conditional
    # net = UNet(num_classes=10, device="cpu", image_size=128)
    print(sum([p.numel() for p in net.parameters()]))
    x = torch.randn(1, 3, 128, 128)
    t = x.new_tensor([500] * x.shape[0]).long()
    y = x.new_tensor([1] * x.shape[0]).long()
    print(net(x, t).shape)
    # print(net(x, t, y).shape)
