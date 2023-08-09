#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Date   : 2023/6/15 17:12
    @Author : chairc
    @Site   : https://github.com/chairc
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class EMA:
    """
    Exponential Moving Average
    """

    def __init__(self, beta):
        """
        Initialize EMA
        :param beta: β
        """
        super().__init__()
        self.beta = beta
        self.step = 0

    def update_model_average(self, ema_model, current_model):
        """
        Update model average
        :param ema_model: EMA model
        :param current_model: Current model
        :return: None
        """
        for current_params, ema_params in zip(current_model.parameters(), ema_model.parameters()):
            old_weight, up_weight = ema_params, current_params.data
            ema_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old_weight, new_weight):
        """
        Update average
        :param old_weight: Old weight
        :param new_weight: New weight
        :return: new_weight or old_weight * self.beta + (1 - self.beta) * new_weight
        """
        if old_weight is None:
            return new_weight
        return old_weight * self.beta + (1 - self.beta) * new_weight

    def step_ema(self, ema_model, model, step_start_ema=2000):
        """
        EMA step
        :param ema_model: EMA model
        :param model: Original model
        :param step_start_ema: Start EMA step
        :return: None
        """
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        """
        Reset parameters
        :param ema_model: EMA model
        :param model: Original model
        :return: None
        """
        ema_model.load_state_dict(model.state_dict())


class SelfAttention(nn.Module):
    """
    SelfAttention block
    """

    def __init__(self, channels, size):
        """
        Initialize the self-attention block
        :param channels: Channels
        :param size: Size
        """
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.size = size
        # batch_first is not supported in pytorch 1.8.
        # If you want to support upgrading to 1.9 and above, or use the following code to transpose
        self.mha = nn.MultiheadAttention(embed_dim=channels, num_heads=4, batch_first=True)
        self.ln = nn.LayerNorm(normalized_shape=[channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm(normalized_shape=[channels]),
            nn.Linear(in_features=channels, out_features=channels),
            nn.GELU(),
            nn.Linear(in_features=channels, out_features=channels),
        )

    def forward(self, x):
        """
        SelfAttention forward
        :param x: Input
        :return: attention_value
        """
        # First perform the shape transformation, and then use 'swapaxes' to exchange the first
        # second dimensions of the new tensor
        x = x.view(-1, self.channels, self.size * self.size).swapaxes(1, 2)
        x_ln = self.ln(x)
        # batch_first is not supported in pytorch 1.8.
        # If you want to support upgrading to 1.9 and above, or use the following code to transpose
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, self.size, self.size)


class DoubleConv(nn.Module):
    """
    Double convolution
    """

    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        """
        Initialize the double convolution block
        :param in_channels: Input channels
        :param out_channels: Output channels
        :param mid_channels: Middle channels
        :param residual: Whether residual
        """
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=1, num_channels=mid_channels),
            nn.GELU(),
            nn.Conv2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=1, num_channels=out_channels),
        )

    def forward(self, x):
        """
        DoubleConv forward
        :param x: Input
        :return: Residual or non-residual results
        """
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)


class DownBlock(nn.Module):
    """
    Downsample block
    """

    def __init__(self, in_channels, out_channels, emb_channels=256):
        """
        Initialize the downsample block
        :param in_channels: Input channels
        :param out_channels: Output channels
        :param emb_channels: Embed channels
        """
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels=in_channels, out_channels=in_channels, residual=True),
            DoubleConv(in_channels=in_channels, out_channels=out_channels),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(in_features=emb_channels, out_features=out_channels),
        )

    def forward(self, x, time):
        """
        DownBlock forward
        :param x: Input
        :param time: Time
        :return: x + emb
        """
        x = self.maxpool_conv(x)
        emb = self.emb_layer(time)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class UpBlock(nn.Module):
    """
    Upsample Block
    """

    def __init__(self, in_channels, out_channels, emb_channels=256):
        """
        Initialize the upsample block
        :param in_channels: Input channels
        :param out_channels: Output channels
        :param emb_channels: Embed channels
        """
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels=in_channels, out_channels=in_channels, residual=True),
            DoubleConv(in_channels=in_channels, out_channels=out_channels, mid_channels=in_channels // 2),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(in_features=emb_channels, out_features=out_channels),
        )

    def forward(self, x, skip_x, time):
        """
        UpBlock forward
        :param x: Input
        :param skip_x: Merged input
        :param time: Time
        :return: x + emb
        """
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(time)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb
