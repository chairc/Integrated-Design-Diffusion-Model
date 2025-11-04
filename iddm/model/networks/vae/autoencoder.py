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
    @Date   : 2025/07/31 9:43
    @Author : chairc
    @Site   : https://github.com/chairc
"""
import torch
import torch.nn as nn
from typing import Tuple
from iddm.config.setting import IMAGE_CHANNEL
from iddm.model.modules.conv import VAEConv2d
from iddm.model.modules.block import VAEResidualBlock, VAEUpBlock


class Autoencoder(nn.Module):
    """
    VAE Implementation for Latent Diffusion Model (1/8 Compression Ratio)
    Peculiarity:
    1. Medium compression ratio: Compress the image to 1/8 size latent space (more detail than 1/16)
    2. It is suitable for scenarios that require high generation quality, and the video memory occupation is moderate
    3. Symmetrical structure design to ensure that the encoder and decoder sizes are strictly matched
    """

    def __init__(
            self,
            in_channels: int = IMAGE_CHANNEL,
            latent_channels: int = 4,
            base_channels: int = 64,
            scale_factor: float = 0.18215,
            act: str = "silu",
            device: str = "cpu"
    ):
        """
        Initialize the Autoencoder model
        :param in_channels: Input image channels, default is 3 (RGB)
        :param latent_channels: Number of latent channels, default is 4
        :param base_channels: Base channel size, default is 64
        :param scale_factor: Latent variable scaling factor, default is 0.18215
        :param act: Activation function, default is "silu"
        :param device: Device type, default is "cpu"
        """
        super(Autoencoder, self).__init__()
        self.in_channels = in_channels
        # The number of latent channels, usually 4
        self.latent_channels = latent_channels
        self.base_channels = base_channels
        # latent variable scaling factor
        self.scale_factor = scale_factor
        self.act = act
        self.device = device

        self.to_latent = nn.Conv2d(base_channels * 4, latent_channels * 2, kernel_size=3, padding=1)

        # Decoder, 3 upsamples to restore the original size
        self.from_latent = nn.Sequential(
            nn.Conv2d(latent_channels, base_channels * 4, kernel_size=3, padding=1),
            nn.GroupNorm(32, base_channels * 4),
            nn.SiLU()
        )

        # Encoder, 3 downsamples Total compression ratio 1/8
        self.encoder = nn.Sequential(
            # Downsample1: [B, C, H, W] [B, 64, H/2, W/2]
            VAEConv2d(in_channels, base_channels, downsample=True, act=act),
            # Downsample2: [B, 128, H/4, W/4]
            VAEConv2d(base_channels, base_channels * 2, downsample=True, act=act),
            VAEResidualBlock(base_channels * 2, base_channels * 2, act=act),
            # Downsample3: [B, 256, H/8, W/8]
            VAEConv2d(base_channels * 2, base_channels * 4, downsample=True, act=act),
            VAEResidualBlock(base_channels * 4, base_channels * 4, act=act),
            # Final feature map: [B, 256, H/8, W/8]
            VAEConv2d(base_channels * 4, base_channels * 4, downsample=False, act=act),
            # Latent space projection: output mean and variance
            self.to_latent
        )

        self.decoder = nn.Sequential(
            self.from_latent,
            # [B, 256, H/8, W/8]
            VAEResidualBlock(base_channels * 4, base_channels * 4, act=act),
            # Upsample1: [B, 128, H/4, W/4]
            VAEUpBlock(base_channels * 4, base_channels * 2, act=act),
            VAEResidualBlock(base_channels * 2, base_channels * 2, act=act),
            # Upsample2: [B, 64, H/2, W/2]
            VAEUpBlock(base_channels * 2, base_channels, act=act),
            VAEResidualBlock(base_channels, base_channels, act=act),
            # Upsample3: [B, C, H, W]
            VAEUpBlock(base_channels, base_channels, act=act),
            nn.Conv2d(base_channels, in_channels, kernel_size=3, padding=1),
            # The output range [-1, 1] matches the diffusion model input
            nn.Tanh()
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode images into subconscious space (1/8 size)
        :param x: Input image [B, C, H, W]
        :return: Mean and logarithmic variance of latent variables [B, L, H/8, W/8]
        """
        # Make sure the input image size is a multiple of 8 (3 downsamples, 2x each time)
        assert x.shape[2] % 8 == 0 and x.shape[3] % 8 == 0, \
            f"The input image size must be a multiple of 8, which is current {x.shape[2:4]}"

        stats = self.encoder(x)
        mean, log_var = torch.chunk(stats, 2, dim=1)
        return mean, log_var

    @staticmethod
    def reparameterize(
            mean: torch.Tensor,
            log_var: torch.Tensor,
            sample: bool = True
    ) -> torch.Tensor:
        """
        Reparameterize sampling, Implement differentiable latent variable sampling
        :param mean: Mean latent variables [B, L, H/8, W/8]
        :param log_var: Logarithmic variance of latent variables [B, L, H/8, W/8]
        :param sample: Whether to sample, True means sampling during training, False means using the mean for inference
        :return: Latent variables after sampling [B, L, H/8, W/8]
        """
        if sample:
            std = torch.exp(0.5 * log_var)
            # Standard normal noise
            eps = torch.randn_like(std)
            return mean + eps * std
        # Use the mean directly when inference
        return mean

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct images from latent variables decoding
        :param z: latent variables [B, L, H/8, W/8]
        :return: reconstruct variables [B, C, H, W]
        """
        # Reverse scaling
        z = z / self.scale_factor
        x = self.decoder(z)
        return x

    def forward(
            self,
            x: torch.Tensor,
            sample: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full forward propagation: encoding->sampling->decoding
        :param x: Input image [B, C, H, W]
        :param sample: Whether to sample, True indicates sampling during training, False indicates sampling
        :return: Reconstructed image [B, C, H, W], Mean latent variables [B, L, H/8, W/8], Logarithmic variance [B, L, H/8, W/8]
        """
        mean, log_var = self.encode(x)
        z = self.reparameterize(mean, log_var, sample)
        # Scaling latent variables
        z_scaled = z * self.scale_factor
        recon_x = self.decode(z_scaled)

        # Verify that the output dimensions are consistent with the input
        if recon_x.shape != x.shape:
            raise ValueError(f"Recon shape {recon_x.shape[2:]} and input shape {x.shape[2:]} mismatch.")

        return recon_x, mean, log_var


if __name__ == "__main__":
    # Init Autoencoder
    model = Autoencoder(in_channels=3, latent_channels=4, base_channels=64)
    print(model)

    # Encoder and decode a sample image
    x = torch.randn(1, 3, 512, 512)
    recon_x, mean, log_var = model(x)

    print(f"Input shape: {x.shape}, Recon shape: {recon_x.shape}, Mean shape: {mean.shape}, Std shape: {log_var.shape}")
