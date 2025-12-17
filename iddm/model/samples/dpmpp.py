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
    @Date   : 2025/09/03 11:20
    @Author : Elizabeth
    @Site   : https://github.com/RachelElizaUK
"""
from typing import Optional, List, Union

import torch
from torch import nn

from tqdm import tqdm

from iddm.model.samples.ddim import DDIMDiffusion
from iddm.utils.logger import get_logger

logger = get_logger(name=__name__)


class DPMPlusPlusDiffusion(DDIMDiffusion):
    """
    DPM++ class
    Implements DPM++ (Denoising Probability Model++) sampling algorithm
    """

    def __init__(
            self,
            noise_steps: int = 1000,
            sample_steps: int = 20,
            beta_start: float = 1e-4,
            beta_end: float = 2e-2,
            img_size: Optional[List[int]] = None,
            device: Union[str, torch.device] = "cpu",
            schedule_name: str = "linear",
            latent: bool = False,
            latent_channel: int = 8,
            autoencoder: Optional[nn.Module] = None,
            order: int = 2  # 2 or 3 for DPM++ 2M or 3M
    ):
        """
        Initialize DPM++ sampler
        :param noise_steps: Total noise steps
        :param sample_steps: Sampling steps (accelerated)
        :param beta_start: β start value
        :param beta_end: β end value
        :param img_size: Image size
        :param device: Device type
        :param schedule_name: Noise schedule type
        :param latent: Whether to use latent diffusion
        :param latent_channel: Latent channel size
        :param autoencoder: Autoencoder model for latent diffusion
        :param order: Solver order (2 or 3)
        """
        super().__init__(noise_steps=noise_steps, sample_steps=sample_steps, beta_start=beta_start, beta_end=beta_end,
                         img_size=img_size, device=device, schedule_name=schedule_name, latent=latent,
                         latent_channel=latent_channel, autoencoder=autoencoder)
        self.order = order
        self.eta = 1.0  # DPM++ uses 1.0 for stochastic sampling by default

    def _sample_loop(
            self,
            model: nn.Module,
            x: Optional[torch.Tensor] = None,
            n: int = 1,
            labels: Optional[torch.Tensor] = None,
            cfg_scale: Optional[float] = None
    ) -> torch.Tensor:
        """
        DPM++ sampling method
        :param model: Model
        :param x: Input image tensor (starting point)
        :param n: Number of samples
        :param labels: Conditional labels
        :param cfg_scale: Classifier-free guidance scale
        :return: Generated images
        """
        logger.info(msg=f"Current order={self.order}")
        # Get time steps in reverse order
        time_steps = list(self.time_step)

        for i, (t, t_prev) in tqdm(enumerate(time_steps), total=len(time_steps)):
            # Prepare time tensors
            t_tensor = (torch.ones(n) * t).long().to(self.device)
            t_prev_tensor = (torch.ones(n) * t_prev).long().to(self.device)

            # Get alpha values
            alpha_t = self.alpha_hat[t_tensor][:, None, None, None]
            alpha_prev = self.alpha_hat[t_prev_tensor][:, None, None, None]

            # Predict noise
            predicted_noise = self._get_predicted_noise(model, x, t_tensor, labels, cfg_scale)

            # Calculate x0 prediction
            if self.latent:
                x = x.clamp(-1, 1)  # Ensure latent values are in range
            x0 = (x - torch.sqrt(1 - alpha_t) * predicted_noise) / torch.sqrt(alpha_t)
            x0 = torch.clamp(x0, -1.0, 1.0)

            # Base coefficients
            sigma = self.eta * torch.sqrt((1 - alpha_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_prev))

            # DPM++ 2M (second-order)
            if self.order == 2:
                if i < len(time_steps) - 1:
                    # Intermediate noise prediction
                    x_inter = torch.sqrt(alpha_prev) * x0 + torch.sqrt(
                        1 - alpha_prev - sigma ** 2) * predicted_noise
                    predicted_noise_inter = self._get_predicted_noise(model, x_inter, t_prev_tensor, labels,
                                                                      cfg_scale)

                    # Second-order correction
                    predicted_noise = (3 * predicted_noise - predicted_noise_inter) / 2

            # DPM++ 3M (third-order)
            elif self.order == 3:
                if i < len(time_steps) - 1:
                    _, t_next = time_steps[i + 1]
                    t_next_tensor = (torch.ones(n) * t_next).long().to(self.device)
                    alpha_next = self.alpha_hat[t_next_tensor][:, None, None, None]

                    # First intermediate step, 1e-8 to avoid NaN
                    sqrt_term1 = torch.sqrt(torch.clamp(1 - alpha_prev - sigma ** 2, min=1e-8))
                    x_inter1 = torch.sqrt(alpha_prev) * x0 + sqrt_term1 * predicted_noise
                    pred_noise1 = self._get_predicted_noise(model, x_inter1, t_prev_tensor, labels, cfg_scale)

                    # Second intermediate step, 1e-8 to avoid NaN
                    sqrt_term2 = torch.sqrt(torch.clamp(1 - alpha_next - sigma ** 2, min=1e-8))
                    x_inter2 = torch.sqrt(alpha_next) * x0 + sqrt_term2 * pred_noise1
                    pred_noise2 = self._get_predicted_noise(model, x_inter2, t_next_tensor, labels, cfg_scale)

                    # Third-order correction
                    predicted_noise = (23 * predicted_noise - 16 * pred_noise1 + 5 * pred_noise2) / 12
                    # Or use a more stable variant
                    # predicted_noise = (18 * predicted_noise - 12 * pred_noise1 + 3 * pred_noise2) / 9

            # Add noise for stochastic sampling
            noise = torch.randn_like(x) if t > 1 else torch.zeros_like(x)

            # Update sample
            sqrt_term = torch.sqrt(torch.clamp(1 - alpha_prev - sigma ** 2, min=1e-8))
            x = torch.sqrt(alpha_prev) * x0 + sqrt_term * predicted_noise + sigma * noise
        return x
