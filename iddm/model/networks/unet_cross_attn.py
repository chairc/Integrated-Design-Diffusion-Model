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
    @Date   : 2025/8/12 16:17
    @Author : chairc
    @Site   : https://github.com/chairc
"""
import torch

from iddm.model.networks.unet import UNet
from iddm.model.modules.attention import CrossAttention


class UNetCrossAttn(UNet):
    """
    UNetCrossAttention
    Replace SelfAttention with CrossAttention
    """

    def __init__(self, **kwargs):
        """
        Initialize the UNetCrossAttention network
        :param in_channel: Input channel
        :param out_channel: Output channel
        :param channel: The list of channel
        :param time_channel: Time channel
        :param num_classes: Number of classes
        :param image_size: Adaptive image size
        :param device: Device type
        :param act: Activation function
        """
        super(UNetCrossAttn, self).__init__(**kwargs)

        # channel: 128
        # size: size / 2
        self.sa1 = CrossAttention(q_channels=self.channel[2], kv_channels=self.channel[2], size=self.image_size_list[1],
                                  act=self.act)
        # channel: 256
        # size: size / 4
        self.sa2 = CrossAttention(q_channels=self.channel[3], kv_channels=self.channel[3], size=self.image_size_list[2],
                                  act=self.act)
        # channel: 256
        # size: size / 8
        self.sa3 = CrossAttention(q_channels=self.channel[3], kv_channels=self.channel[3], size=self.image_size_list[3],
                                  act=self.act)

        # channel: 128
        # size: size / 4
        self.sa4 = CrossAttention(q_channels=self.channel[2], kv_channels=self.channel[2], size=self.image_size_list[2],
                                  act=self.act)
        # channel: 128
        # size: size / 2
        self.sa5 = CrossAttention(q_channels=self.channel[1], kv_channels=self.channel[1], size=self.image_size_list[1],
                                  act=self.act)
        # channel: 128
        # size: size
        self.sa6 = CrossAttention(q_channels=self.channel[1], kv_channels=self.channel[1], size=self.image_size_list[0],
                                  act=self.act)

    def forward(self, x, time, y=None):
        """
        Forward
        :param x: Input
        :param time: Time
        :param y: Input label
        :return: output
        """
        time = self.encode_time_with_label(time=time, y=y)

        x1 = self.inc(x)
        x2 = self.down1(x1, time)
        x2_sa = self.sa1(x2, x2)
        x3 = self.down2(x2_sa, time)
        x3_sa = self.sa2(x3, x3)
        x4 = self.down3(x3_sa, time)
        x4_sa = self.sa3(x4, x4)

        bot1_out = self.bot1(x4_sa)
        bot2_out = self.bot2(bot1_out)
        bot3_out = self.bot3(bot2_out)

        up1_out = self.up1(bot3_out, x3_sa, time)
        up1_sa_out = self.sa4(up1_out, up1_out)
        up2_out = self.up2(up1_sa_out, x2_sa, time)
        up2_sa_out = self.sa5(up2_out, up2_out)
        up3_out = self.up3(up2_sa_out, x1, time)
        up3_sa_out = self.sa6(up3_out, up3_out)
        output = self.outc(up3_sa_out)
        return output


if __name__ == "__main__":
    # Unconditional
    net = UNetCrossAttn(device="cpu", image_size=[64, 64])
    # Conditional
    # net = UNet(num_classes=10, device="cpu", image_size=128)
    print(sum([p.numel() for p in net.parameters()]))
    x = torch.randn(1, 3, 64, 64)
    t = x.new_tensor([500] * x.shape[0]).long()
    y = x.new_tensor([1] * x.shape[0]).long()
    print(net(x, t).shape)
    # print(net(x, t, y).shape)
