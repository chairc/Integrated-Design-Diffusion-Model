#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Date   : 2023/6/23 22:26
    @Author : chairc
    @Site   : https://github.com/chairc
"""

from iddm.model.networks.unet import UNet


class UNetSlim(UNet):
    """
    UNet-Slim
    This is a slim network demo, reduce 45% GPU used
    """

    def __init__(self, **kwargs):
        """
        Initialize the UNet-Slim network
        :param in_channel: Input channel
        :param out_channel: Output channel
        :param channel: The list of channel
        :param time_channel: Time channel
        :param num_classes: Number of classes
        :param image_size: Adaptive image size
        :param device: Device type
        :param act: Activation function
        """
        super(UNetSlim, self).__init__(**kwargs)

    def forward(self, x, time, y=None):
        """
        Forward
        :param x: Input
        :param time: Time
        :param y: Input label
        :return: output
        """
        time = self.encode_time_with_label(time=time, y=y)

        x = self.inc(x)
        x1 = x
        x = self.down1(x, time)
        x = self.sa1(x)
        x2_sa = x
        x = self.down2(x, time)
        x3_sa = x
        x = self.down3(x, time)
        x = self.sa3(x)

        x = self.bot1(x)
        x = self.bot2(x)
        x = self.bot3(x)

        x = self.up1(x, x3_sa, time)
        x = self.up2(x, x2_sa, time)
        x = self.sa5(x)
        x = self.up3(x, x1, time)
        output = self.outc(x)
        return output
