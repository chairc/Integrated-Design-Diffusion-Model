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
    @Date   : 2025/8/12 13:34
    @Author : chairc
    @Site   : https://github.com/chairc
"""
import torch

from iddm.model.networks.unet import UNet
from iddm.model.modules.attention import FlashSelfAttention


class UNetFlashSelfAttn(UNet):
    """
    UNetFlashSelfAttn
    Replace SelfAttention with FlashSelfAttention
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
        super(UNetFlashSelfAttn, self).__init__(**kwargs)

        # channel: 128
        # size: size / 2
        self.sa1 = FlashSelfAttention(channels=self.channel[2], size=self.image_size_list[1], act=self.act)
        # channel: 256
        # size: size / 4
        self.sa2 = FlashSelfAttention(channels=self.channel[3], size=self.image_size_list[2], act=self.act)
        # channel: 256
        # size: size / 8
        self.sa3 = FlashSelfAttention(channels=self.channel[3], size=self.image_size_list[3], act=self.act)

        # channel: 128
        # size: size / 4
        self.sa4 = FlashSelfAttention(channels=self.channel[2], size=self.image_size_list[2], act=self.act)
        # channel: 128
        # size: size / 2
        self.sa5 = FlashSelfAttention(channels=self.channel[1], size=self.image_size_list[1], act=self.act)
        # channel: 128
        # size: size
        self.sa6 = FlashSelfAttention(channels=self.channel[1], size=self.image_size_list[0], act=self.act)


if __name__ == "__main__":
    # Unconditional
    net = UNetFlashSelfAttn(device="cuda:0", image_size=[64, 64])
    net = net.to("cuda:0")
    # Conditional
    # net = UNet(num_classes=10, device="cpu", image_size=128)
    print(sum([p.numel() for p in net.parameters()]))
    x = torch.randn(1, 3, 64, 64).to("cuda:0")
    t = x.new_tensor([500] * x.shape[0]).long().to("cuda:0")
    y = x.new_tensor([1] * x.shape[0]).long().to("cuda:0")
    print(net(x, t).shape)
    # print(net(x, t, y).shape)
