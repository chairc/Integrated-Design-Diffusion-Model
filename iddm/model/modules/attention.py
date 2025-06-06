#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Date   : 2023/12/5 10:19
    @Author : chairc
    @Site   : https://github.com/chairc
"""
import torch.nn as nn
from iddm.model.modules.activation import get_activation_function


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
        x = x.view(-1, self.channels, self.size[0] * self.size[1]).swapaxes(1, 2)
        x_ln = self.ln(x)
        # batch_first is not supported in pytorch 1.8.
        # If you want to support upgrading to 1.9 and above, or use the following code to transpose
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, self.size[0], self.size[1])


class SelfAttentionAD(nn.Module):
    """
    Adaptive head count SelfAttention block
    """

    def __init__(self, channels, size, act="silu", dropout=0.1):
        """
        Initialize the adaptive head count self-attention block
        :param channels: Channels
        :param size: Size
        :param act: Activation function
        """
        super(SelfAttentionAD, self).__init__()
        self.channels = channels
        self.size = size
        self.dropout = dropout

        # Adaptive head count
        head_count = max(1, channels // 64)

        # batch_first is not supported in pytorch 1.8.
        # If you want to support upgrading to 1.9 and above, or use the following code to transpose
        self.mha = nn.MultiheadAttention(embed_dim=channels, num_heads=head_count, batch_first=True)
        self.ln = nn.LayerNorm(normalized_shape=[channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm(normalized_shape=[channels]),
            nn.Linear(in_features=channels, out_features=channels),
            get_activation_function(name=act),
            nn.Dropout(dropout),
            nn.Linear(in_features=channels, out_features=channels),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        """
        SelfAttention forward
        :param x: Input
        :return: attention_value
        """
        batch, channels, height, width = x.shape
        assert height == self.size[0] and width == self.size[1], \
            f"Input size {height}x{width} does not match the expected size {self.size[0]}x{self.size[1]}"
        # Flatten the spatial dimension into sequence dimensions
        # (batch, channels, height*width) -> (batch, seq_len, channels)
        x_flat = x.flatten(2).swapaxes(1, 2)

        # First residual calculation
        x_ln = self.ln(x_flat)
        # batch_first is not supported in pytorch 1.8.
        # If you want to support upgrading to 1.9 and above, or use the following code to transpose
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x_flat

        # Second residual calculation
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(1, 2).view(batch, channels, height, width)
