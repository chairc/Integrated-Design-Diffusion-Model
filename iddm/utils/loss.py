#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Date   : 2025/7/31 10:38
    @Author : chairc
    @Site   : https://github.com/chairc
"""
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MSEKLLoss(nn.Module):
    """
    Combined loss function: MSE reconstruction loss + KL divergence regularization
    Suitable for training VAE models
    """

    def __init__(self, kl_weight: float = 0.0001):
        """
        Initialize the loss function
        :param kl_weight: Weight coefficients of KL divergence, balancing reconstruction loss and KL divergence
        """
        super().__init__()
        self.kl_weight = kl_weight

    def forward(
            self,
            output: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
            x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculate the portfolio loss
        :param output: Output tuples containing reconstructed images, means, and logarithmic variance
        :param x: Original input image [B, C, H, W]
        :return: Total loss (MSE loss + KL divergence * weight)
        """
        recon_x, mean, log_var = output
        # Calculate MSE reconstruction losses
        recon_loss = F.mse_loss(recon_x, x)

        # Calculate KL divergence
        # KL(q(z|x) || p(z)) = -0.5 * sum(1 +  log_var - mean^2 - exp( log_var))
        kl_loss = -0.5 * torch.sum(1 + log_var - mean ** 2 - log_var.exp())
        # Normalize to the number of pixels in the input image
        kl_loss = kl_loss / x.numel()

        # Total loss
        total_loss = recon_loss + self.kl_weight * kl_loss
        return total_loss
