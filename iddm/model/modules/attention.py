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
    @Date   : 2023/12/5 10:19
    @Author : chairc
    @Site   : https://github.com/chairc
"""
import torch
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


class CrossAttention(nn.Module):
    """
    CrossAttention block
    Cross-attention module, Query and Key/Value from different input sources
    """

    def __init__(self, q_channels, kv_channels, size, act="silu"):
        """
        Initialize the Cross Attention module
        :param q_channels: The number of channels for the query
        :param kv_channels: Number of channels for Key and Value
        :param size: Space size (height, width)
        :param act: Activate the function
        """
        super(CrossAttention, self).__init__()
        # Number of Query channels
        self.q_channels = q_channels
        # Key/Value channels
        self.kv_channels = kv_channels
        # Space dimensions (need to match input)
        self.size = size

        # Long cross-attention: Query comes from the main input, and Key/Value comes from the cross-input
        self.mha = nn.MultiheadAttention(embed_dim=q_channels, num_heads=4, kdim=kv_channels, vdim=kv_channels,
                                         batch_first=True)

        # Normalization layer (normalizes Query and Key/Value respectively)
        self.ln_q = nn.LayerNorm(normalized_shape=[q_channels])
        self.ln_kv = nn.LayerNorm(normalized_shape=[kv_channels])

        # Feedforward Network (Consistent with SelfAttention Structure)
        self.ff_self = nn.Sequential(
            nn.LayerNorm(normalized_shape=[q_channels]),
            nn.Linear(in_features=q_channels, out_features=q_channels),
            get_activation_function(name=act),
            nn.Linear(in_features=q_channels, out_features=q_channels),
        )

    def forward(self, x_q, x_kv):
        """
        Forward propagation
        :param x_q: As input to Query, shape [batch, q_channels, height, width]
        :param x_kv: As input for Key and Value, shape [batch, kv_channels, height, width]
        :return: The fused feature is [batch, q_channels, height, width]
        """
        # Verify the input dimensions
        batch, q_c, h, w = x_q.shape
        assert (h, w) == (self.size[0],self.size[1]), f"Query size {h}x{w} does not match the expected {self.size}"

        kv_batch, kv_c, kv_h, kv_w = x_kv.shape
        assert kv_batch == batch, "Query does not match the batch size of Key/Value"
        assert (kv_h, kv_w) == (self.size[0],self.size[1]), f"Query size {kv_h}x{kv_w} does not match the expected {self.size}"
        assert kv_c == self.kv_channels, f"Query channel {kv_c} does not match the expected {self.kv_channels}"

        # Flattening the Dimension of Space: [batch, channels, h*w] -> [batch, seq_len, channels]
        x_q_flat = x_q.view(batch, q_c, h * w).swapaxes(1, 2)
        x_kv_flat = x_kv.view(kv_batch, kv_c, kv_h * kv_w).swapaxes(1, 2)

        # Normalization
        x_q_ln = self.ln_q(x_q_flat)
        x_kv_ln = self.ln_kv(x_kv_flat)

        # Cross-attention calculation: Query is from x_q, Key/Value is from x_kv
        attn_output, _ = self.mha(
            query=x_q_ln,
            key=x_kv_ln,
            value=x_kv_ln
        )

        # Residual join (summed with the original Query input)
        attn_output = attn_output + x_q_flat

        # Feedforward network + residual connection
        attn_output = self.ff_self(attn_output) + attn_output

        # Restore the spatial dimension
        return attn_output.swapaxes(2, 1).view(batch, q_c, h, w)


class FlashSelfAttention(nn.Module):
    """
    FlashAttention-based self-attention module
    It is suitable for replacing the original SelfAttention and improving the computing efficiency in long sequence scenarios
    """

    def __init__(self, channels, size, act="silu", dropout=0.1, dtype=torch.bfloat16):
        """
        Initialize the Flash Self-Attention module
        :param channels: Channels
        :param size: Size of the spatial dimension (height, width)
        :param act: Activation function
        :param dropout: Dropout rate
        :param dtype: Data type for computation, default is bfloat16 for FlashAttention
        """
        super().__init__()
        self.channels = channels
        # (height, width)
        self.size = size
        self.dropout = dropout
        self.dtype = dtype

        # Adaptive Head Count (Consistent with the original logic)
        self.num_heads = max(1, channels // 64)
        assert channels % self.num_heads == 0, "The number of channels must be divisible by the number of heads"
        self.head_dim = channels // self.num_heads

        # QKV projection (merged into one linear layer for efficiency)
        self.qkv_proj = nn.Linear(channels, 3 * channels, dtype=self.dtype)
        self.out_proj = nn.Linear(channels, channels, dtype=self.dtype)

        # Normalization and feedforward networks
        self.ln = nn.LayerNorm(channels, dtype=self.dtype)
        self.ff_self = nn.Sequential(
            nn.LayerNorm(channels, dtype=self.dtype),
            nn.Linear(channels, channels, dtype=self.dtype),
            get_activation_function(act),
            nn.Dropout(dropout),
            nn.Linear(channels, channels, dtype=self.dtype),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        """
        Forward propagation for Flash Self-Attention
        Input shape: [batch, channels, height, width]
        Output shape: [batch, channels, height, width]
        :param x: Input tensor
        :return: Output tensor after self-attention and feedforward network
        """
        batch, c, h, w = x.shape
        assert (h, w) == (self.size[0], self.size[1]), f"Query size {h}x{w} does not match the expected {self.size}"

        # Save the raw data type for output
        x_dtype = x.dtype
        # Convert to bfloat16 for FlashAttention
        x = x.to(self.dtype, non_blocking=True)

        # Flattening the Dimension of Space: [batch, seq_len, channels]，seq_len = h*w
        x_flat = x.flatten(2).transpose(1, 2)  # [B, seq_len, C]
        x_ln = self.ln(x_flat)

        # QKV projection and splitting: [B, seq_len, 3*C] -> [B, seq_len, 3, num_heads, head_dim]
        qkv = self.qkv_proj(x_ln)
        qkv = qkv.view(batch, -1, 3, self.num_heads, self.head_dim)
        qkv = qkv.to(self.dtype, non_blocking=True)

        # FlashAttention calculation (self-attention, Q=K=V)
        # Output: [B, seq_len, num_heads, head_dim]
        from flash_attn import flash_attn_qkvpacked_func
        attn_output = flash_attn_qkvpacked_func(
            qkv,
            dropout_p=self.dropout if self.training else 0.0,
            causal=False
        )  # [B, seq_len, num_heads, head_dim]

        # Merge headers and project them: [B, seq_len, C]
        attn_output = attn_output.view(batch, -1, self.channels)
        attn_output = self.out_proj(attn_output)

        # Residual connection + feedforward network
        attn_output = attn_output + x_flat
        attn_output = self.ff_self(attn_output) + attn_output

        # Restore the spatial dimension
        attn_output = attn_output.to(x_dtype, non_blocking=True)
        return attn_output.transpose(1, 2).view(batch, c, h, w)
