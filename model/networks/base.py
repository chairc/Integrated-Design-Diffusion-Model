#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Date   : 2023/11/8 22:44
    @Author : chairc
    @Site   : https://github.com/chairc
"""
import torch
import torch.nn as nn


class BaseNet(nn.Module):
    """
    Base Network
    """

    def __init__(self, in_channel=3, out_channel=3, channel=None, time_channel=256, num_classes=None, image_size=64,
                 device="cpu", act="silu"):
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
        self.init_channel(channel)
        self.time_channel = time_channel
        self.num_classes = num_classes
        self.image_size = image_size
        self.device = device
        self.act = act

        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_embeddings=self.num_classes, embedding_dim=self.time_channel)

    def init_channel(self, channel):
        """
        Init channel
        If channel is None, this function would set a default channel.
        :param channel: Channel
        :return: global self.channel
        """
        if channel is None:
            self.channel = [32, 64, 128, 256, 512, 1024]
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
