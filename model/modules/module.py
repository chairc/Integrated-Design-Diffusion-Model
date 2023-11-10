#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Date   : 2023/6/15 17:12
    @Author : chairc
    @Site   : https://github.com/chairc
"""
import logging
import coloredlogs

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)
coloredlogs.install(level="INFO")


def get_activation_function(name="silu", inplace=False):
    """
    Get activation function
    :param name: Activation function name
    :param inplace: can optionally do the operation in-place
    :return Activation function
    """
    if name == "relu":
        act = nn.ReLU(inplace=inplace)
    elif name == "relu6":
        act = nn.ReLU6(inplace=inplace)
    elif name == "silu":
        act = nn.SiLU(inplace=inplace)
    elif name == "lrelu":
        act = nn.LeakyReLU(0.1, inplace=inplace)
    elif name == "gelu":
        act = nn.GELU()
    else:
        logger.warning(msg=f"Unsupported activation function type: {name}")
        act = nn.SiLU(inplace=inplace)
    return act


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

    def __init__(self, channels, size, act="silu"):
        """
        Initialize the self-attention block
        :param channels: Channels
        :param size: Size
        :param act: Activation function
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
            get_activation_function(name=act),
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

    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False, act="silu"):
        """
        Initialize the double convolution block
        :param in_channels: Input channels
        :param out_channels: Output channels
        :param mid_channels: Middle channels
        :param residual: Whether residual
        :param act: Activation function
        """
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.act = act
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=1, num_channels=mid_channels),
            get_activation_function(name=self.act),
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
            out = x + self.double_conv(x)
            if self.act == "relu":
                return F.relu(out)
            elif self.act == "relu6":
                return F.relu6(out)
            elif self.act == "silu":
                return F.silu(out)
            elif self.act == "lrelu":
                return F.leaky_relu(out)
            elif self.act == "gelu":
                return F.gelu(out)
            else:
                logger.warning(msg=f"Unsupported activation function type: {self.act}")
                return F.silu(out)
        else:
            return self.double_conv(x)


class DownBlock(nn.Module):
    """
    Downsample block
    """

    def __init__(self, in_channels, out_channels, emb_channels=256, act="silu"):
        """
        Initialize the downsample block
        :param in_channels: Input channels
        :param out_channels: Output channels
        :param emb_channels: Embed channels
        :param act: Activation function
        """
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels=in_channels, out_channels=in_channels, residual=True, act=act),
            DoubleConv(in_channels=in_channels, out_channels=out_channels, act=act),
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

    def __init__(self, in_channels, out_channels, emb_channels=256, act="silu"):
        """
        Initialize the upsample block
        :param in_channels: Input channels
        :param out_channels: Output channels
        :param emb_channels: Embed channels
        :param act: Activation function
        """
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels=in_channels, out_channels=in_channels, residual=True, act=act),
            DoubleConv(in_channels=in_channels, out_channels=out_channels, mid_channels=in_channels // 2, act=act),
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


class BaseConv(nn.Module):
    """
    Base convolution
    Conv2d -> BatchNorm -> Activation function block
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, groups=1, bias=False, act="silu"):
        """
        Initialize the Base convolution
        :param in_channels: Input channels
        :param out_channels: Output channels
        :param kernel_size: Kernel size
        :param stride: Stride
        :param groups: Groups
        :param bias: Bias
        :param act: Activation function
        """
        super().__init__()
        # Same padding
        pad = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              stride=stride, padding=pad, groups=groups, bias=bias)
        self.gn = nn.GroupNorm(num_groups=1, num_channels=out_channels)
        self.act = get_activation_function(name=act, inplace=True)

    def forward(self, x):
        return self.act(self.gn(self.conv(x)))


class Bottleneck(nn.Module):
    """
    Standard bottleneck
    """

    def __init__(self, in_channels, out_channels, shortcut=True, expansion=0.5, act="silu"):
        """
        Initialize the Bottleneck
        :param in_channels: Input channels
        :param out_channels: Output channels
        :param shortcut: Shortcut, such as x + y
        :param expansion: Factor
        :param act: Activation function
        """
        super().__init__()
        mid_channels = int(out_channels * expansion)
        self.conv1 = BaseConv(in_channels=in_channels, out_channels=mid_channels, kernel_size=1, stride=1, act=act)
        self.conv2 = BaseConv(in_channels=mid_channels, out_channels=out_channels, kernel_size=3, stride=1, act=act)
        # Use shortcut
        self.use_add = shortcut and in_channels == out_channels

    def forward(self, x):
        y = self.conv2(self.conv1(x))

        if self.use_add:
            y = y + x
        return y


class SPPFBottleneck(nn.Module):
    """
    SPPF Bottleneck
    https://github.com/ultralytics/yolov5/blob/master/models/common.py
    https://github.com/ultralytics/yolov5/blob/3eefab1bb109214a614485b6c5f80f22c122f2b2/models/common.py#L182
    """

    # 'kernel_sizes = 5' is the same as 'SPP(kernel_sizes=(5, 9, 13))'
    def __init__(self, in_channels, out_channels, kernel_size=5, act="silu"):
        """
        Initialize the SPPFBottleneck
        :param in_channels: Input channels
        :param out_channels: Output channels
        :param kernel_size: Kernel size
        :param act: Activation function
        """
        super().__init__()
        mid_channels = in_channels // 2
        self.conv1 = BaseConv(in_channels=in_channels, out_channels=mid_channels, kernel_size=1, stride=1, act=act)
        self.conv2 = BaseConv(in_channels=mid_channels * 4, out_channels=out_channels, kernel_size=1, stride=1, act=act)
        self.m = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

    def forward(self, x):
        x = self.conv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.conv2(torch.cat([x, y1, y2, self.m(y2)], dim=1))


class CSPLayer(nn.Module):
    """
    CSP Bottleneck with 3 convolutions
    """

    def __init__(self, in_channels, out_channels, n=1, shortcut=True, expansion=0.5, act="silu"):
        """
        Initialize the CSPLayer
        :param in_channels: Input channels
        :param out_channels: Output channels
        :param n: Number of Bottlenecks
        :param shortcut: Shortcut, such as x + y
        :param expansion: Factor
        :param act: Activation function
        """
        super().__init__()
        mid_channels = int(out_channels * expansion)  # hidden channels
        self.conv1 = BaseConv(in_channels=in_channels, out_channels=mid_channels, kernel_size=1, stride=1, act=act)
        self.conv2 = BaseConv(in_channels=in_channels, out_channels=mid_channels, kernel_size=1, stride=1, act=act)
        self.conv3 = BaseConv(in_channels=2 * mid_channels, out_channels=out_channels, kernel_size=1, stride=1, act=act)
        module_list = [
            Bottleneck(in_channels=mid_channels, out_channels=mid_channels, shortcut=shortcut, expansion=1.0,
                       act=act)
            for _ in range(n)
        ]
        self.m = nn.Sequential(*module_list)

    def forward(self, x):
        x_1 = self.conv1(x)
        x_2 = self.conv2(x)
        x_1 = self.m(x_1)
        x = torch.cat([x_1, x_2], dim=1)
        return self.conv3(x)


class CSPDarkDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, emb_channels=256, n=1, act="silu"):
        super().__init__()
        self.conv_csp = nn.Sequential(
            BaseConv(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2, act=act),
            CSPLayer(in_channels=out_channels, out_channels=out_channels, n=n, act=act)
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(in_features=emb_channels, out_features=out_channels),
        )

    def forward(self, x, time):
        x = self.conv_csp(x)
        emb = self.emb_layer(time)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class CSPDarkUpBlock(nn.Module):

    def __init__(self, in_channels, out_channels, emb_channels=256, n=1, act="silu"):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = BaseConv(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, act=act)
        self.csp = CSPLayer(in_channels=in_channels, out_channels=out_channels, n=n, shortcut=False, act=act)

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(in_features=emb_channels, out_features=out_channels),
        )

    def forward(self, x, skip_x, time):
        x = self.conv(x)
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(time)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb
