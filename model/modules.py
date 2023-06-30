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
    指数移动平均
    """

    def __init__(self, beta):
        """
        初始化EMA
        :param beta: β
        """
        super().__init__()
        self.beta = beta
        self.step = 0

    def update_model_average(self, ema_model, current_model):
        """
        更新模型均值
        :param ema_model: EMA模型
        :param current_model: 当前模型
        :return: None
        """
        for current_params, ema_params in zip(current_model.parameters(), ema_model.parameters()):
            old_weight, up_weight = ema_params, current_params.data
            ema_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old_weight, new_weight):
        """
        更新均值
        :param old_weight: 旧权重
        :param new_weight: 新权重
        :return: new_weight或old_weight * self.beta + (1 - self.beta) * new_weight
        """
        if old_weight is None:
            return new_weight
        return old_weight * self.beta + (1 - self.beta) * new_weight

    def step_ema(self, ema_model, model, step_start_ema=2000):
        """
        EMA步长
        :param ema_model: EMA模型
        :param model: 原模型
        :param step_start_ema: 开始 EMA步长
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
        重置参数
        :param ema_model: EMA模型
        :param model: 原模型
        :return: None
        """
        ema_model.load_state_dict(model.state_dict())


class SelfAttention(nn.Module):
    """
    自注意力模块
    """

    def __init__(self, channels, size):
        """
        初始化自注意力块
        :param channels: 通道
        :param size: 尺寸
        """
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.size = size
        # pytorch1.8中不支持batch_first，若要支持升级为1.9及以上，或使用一下代码进行转置
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
        前向传播
        :param x: 输入
        :return: attention_value
        """
        # 首先进行形状变换，再用swapaxes对新张量的第1和2维度进行交换
        x = x.view(-1, self.channels, self.size * self.size).swapaxes(1, 2)
        x_ln = self.ln(x)
        # pytorch1.8中不支持batch_first，若要支持升级为1.9及以上，或使用一下代码进行转置
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, self.size, self.size)


class DoubleConv(nn.Module):
    """
    双卷积
    """

    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        """
        初始化双卷积
        :param in_channels: 输入通道
        :param out_channels: 输出通道
        :param mid_channels: 中间通道
        :param residual: 是否残差
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
        前向传播
        :param x: 输入
        :return: 残差结果或非残差结果
        """
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)


class DownBlock(nn.Module):
    """
    下采样块
    """

    def __init__(self, in_channels, out_channels, emb_channels=256):
        """
        初始化下采样块
        :param in_channels: 输入通道
        :param out_channels: 输出通道
        :param emb_channels: 嵌入通道
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
        前向传播
        :param x: 输入
        :param time: 时间
        :return: x + emb
        """
        x = self.maxpool_conv(x)
        emb = self.emb_layer(time)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class UpBlock(nn.Module):
    """
    上采样块
    """

    def __init__(self, in_channels, out_channels, emb_channels=256):
        """
        初始化上采样块
        :param in_channels: 输入通道
        :param out_channels: 输出通道
        :param emb_channels: 嵌入通道
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
        前向传播
        :param x: 输入
        :param skip_x: 需要合并的输入
        :param time: 时间
        :return: x + emb
        """
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(time)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb
