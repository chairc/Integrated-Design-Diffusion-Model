#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Date   : 2023/11/8 22:44
    @Author : chairc
    @Site   : https://github.com/chairc
"""
import torch
import torch.nn as nn

from config.setting import IMAGE_CHANNEL, TIME_CHANNEL, CHANNEL_LIST, DEFAULT_IMAGE_SIZE


class BaseNet(nn.Module):
    """
    Base Network
    """

    def __init__(self, in_channel=IMAGE_CHANNEL, out_channel=IMAGE_CHANNEL, channel=None, time_channel=TIME_CHANNEL,
                 num_classes=None, image_size=None, device="cpu", act="silu"):
        """
        Initialize the Base network
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
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.channel = None
        self.init_channel(channel=channel)
        self.time_channel = time_channel
        self.num_classes = num_classes
        self.image_size = None
        self.init_image_size(image_size=image_size)
        self.device = device
        self.act = act

        # Init image size list
        self.image_size_list = []
        self.init_image_size_list()

        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_embeddings=self.num_classes, embedding_dim=self.time_channel)

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
