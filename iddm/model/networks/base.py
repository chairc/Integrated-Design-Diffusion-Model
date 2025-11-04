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
    @Date   : 2023/11/8 22:44
    @Author : chairc
    @Site   : https://github.com/chairc
"""
import torch
import torch.nn as nn

from iddm.config.setting import IMAGE_CHANNEL, TIME_CHANNEL, CHANNEL_LIST, DEFAULT_IMAGE_SIZE
from iddm.model.networks.conditional import TextConditionAdapter, ClassConditionAdapter


class BaseNet(nn.Module):
    """
    Base Network
    """

    def __init__(self, mode="class", in_channel=IMAGE_CHANNEL, out_channel=IMAGE_CHANNEL, channel=None,
                 time_channel=TIME_CHANNEL, image_size=None, device="cpu", act="silu", **kwargs):
        """
        Initialize the Base network
        :param mode: Conditional Guidance Mode
        :param in_channel: Input channel
        :param out_channel: Output channel
        :param channel: The list of channel
        :param time_channel: Time channel
        :param image_size: Adaptive image size
        :param device: Device type
        :param act: Activation function
        """
        super().__init__()
        self.mode = mode
        self.label_emb = None
        self.condition_adapter = None
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.channel = None
        self.init_channel(channel=channel)
        self.time_channel = time_channel
        self.image_size = None
        self.init_image_size(image_size=image_size)
        self.device = device
        self.act = act
        self.kwargs = kwargs

        # Get parameters
        self.num_classes = self.kwargs.get("num_classes", None)
        self.text = self.kwargs.get("text", None)

        # Conditional adapter initialization
        self.init_conditional_adapter()

        # Init image size list
        self.image_size_list = []
        self.init_image_size_list()

    def init_channel(self, channel):
        """
        Init channel
        If channel is None, this function would set a default channel.
        :param channel: Channel
        :return: global self.channel
        """
        if channel is None or not isinstance(channel, list):
            self.channel = CHANNEL_LIST
        else:
            self.channel = channel

    def init_conditional_adapter(self):
        """
        Conditional adapter initialization
        """
        if self.mode == "text" and self.text is not None:
            self.condition_adapter = TextConditionAdapter(emb_channel=self.time_channel, device=self.device)
        elif self.mode == "class" and self.num_classes is not None:
            # TODO: Add and replace ClassConditionAdapter
            self.condition_adapter = ClassConditionAdapter(num_classes=self.num_classes, emb_channel=self.time_channel)
        else:
            self.condition_adapter = None

    def pos_encoding(self, time, channels):
        """
        Base network position encoding
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

    def init_image_size(self, image_size):
        """
        Init image size
        :param image_size: Image size
        """
        if image_size is None:
            self.image_size = DEFAULT_IMAGE_SIZE
        else:
            self.image_size = image_size

    def init_image_size_list(self):
        """
        Init image size list
        :return: global self.image_size_list
        """
        # Create image size list
        try:
            h, w = self.image_size
            new_image_size_list = [[h, w], [h / 2, w / 2], [h / 4, w / 4], [h / 8, w / 8], [h / 16, w / 16],
                                   [h / 32, w / 32]]
            self.image_size_list = [[int(size_h), int(size_w)] for size_h, size_w in new_image_size_list]
        except Exception:
            raise IndexError("The image size is set too small and the preprocessing exceeds the index range. "
                             "It is recommended that the image length and width be set to no less than 32.")

    def encode_time_with_label(self, time, y):
        """
        Encode time with label
        :param time: Time
        :param y: Input label (class or text)
        :return: time
        """
        time = time.unsqueeze(-1).type(torch.float)
        time = self.pos_encoding(time, self.time_channel)

        if y is not None:
            time += self.condition_adapter(y)
        return time
