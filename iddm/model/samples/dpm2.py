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
    @Date   : 2025/9/2 16:18
    @Author : Elizabeth
    @Site   : https://github.com/RachelElizaUK
"""
from typing import Optional, List, Union

import torch
from torch import nn

from tqdm import tqdm

from iddm.model.samples.base import BaseDiffusion
from iddm.utils.logger import get_logger

logger = get_logger(name=__name__)


class DPM2Diffusion(BaseDiffusion):
    """
    DPM2 sampler class
    """

    def __init__(
            self,
            noise_steps: int = 1000,
            sample_steps: int = 50,
            beta_start: float = 1e-4,
            beta_end: float = 2e-2,
            img_size: Optional[List[int]] = None,
            device: Union[str, torch.device] = "cpu",
            schedule_name: str = "linear",
            latent: bool = False,
            latent_channel: int = 8,
            autoencoder: Optional[nn.Module] = None
    ):
        """
        The implement of DPM2
        :param noise_steps: Total noise steps
        :param sample_steps: Sampling steps (DPM2 typically uses fewer steps than full noise steps)
        :param beta_start: β start value
        :param beta_end: β end value
        :param img_size: Image size
        :param device: Device type
        :param schedule_name: Noise schedule name
        :param latent: Whether to use latent diffusion
        :param latent_channel: Latent channel size
        :param autoencoder: Autoencoder model for latent diffusion
        """
        super().__init__(
            noise_steps=noise_steps,
            beta_start=beta_start,
            beta_end=beta_end,
            img_size=img_size,
            device=device,
            schedule_name=schedule_name,
            latent=latent,
            latent_channel=latent_channel,
            autoencoder=autoencoder
        )
        self.sample_steps = sample_steps
        self.eta = 0.0  # DPM2 uses deterministic path by default
        self._init_time_steps()

    def _init_time_steps(self):
        """
        Initialize time steps for DPM2 sampling
        Creates a sequence of time steps from T down to 0 with equal intervals
        """
        step_ratio = self.noise_steps // self.sample_steps
        self.time_steps = torch.arange(0, self.noise_steps, step_ratio).long() + 1
        self.time_steps = reversed(torch.cat((torch.tensor([0], dtype=torch.long), self.time_steps)))
        self.time_steps = list(zip(self.time_steps[:-1], self.time_steps[1:]))

    def _sample_loop(
            self,
            model: nn.Module,
            x: Optional[torch.Tensor] = None,
            n: int = 1,
            labels: Optional[torch.Tensor] = None,
            cfg_scale: Optional[float] = None
    ) -> torch.Tensor:
        """
        DPM2 sample loop method
        :param model: Diffusion model
        :param x: Initial input tensor (optional)
        :param n: Number of samples to generate
        :param labels: Conditional labels (optional)
        :param cfg_scale: Classifier-free guidance scale (optional)
        :return: Generated images tensor
        """
        for i, p_i in tqdm(self.time_steps, position=0, total=len(self.time_steps)):
            # Current and previous time steps
            t = (torch.ones(n) * i).long().to(self.device)
            p_t = (torch.ones(n) * p_i).long().to(self.device)

            # Get alpha values for current and previous steps
            alpha_curr = self.alpha_hat[t][:, None, None, None]
            alpha_prev = self.alpha_hat[p_t][:, None, None, None]

            # Step 1: Predict noise at current time step
            predicted_noise = self._get_predicted_noise(model, x, t, labels, cfg_scale)

            # Step 2: Compute x0 from current x and predicted noise
            x0 = (x - torch.sqrt(1 - alpha_curr) * predicted_noise) / torch.sqrt(alpha_curr)
            x0 = torch.clamp(x0, -1.0, 1.0)  # Stabilize x0 prediction

            # Step 3: Midpoint prediction (DPM2 uses 2nd-order method)
            t_mid = (t + p_t) // 2
            alpha_mid = self.alpha_hat[t_mid][:, None, None, None]

            # Compute midpoint x
            x_mid = torch.sqrt(alpha_mid) * x0 + torch.sqrt(1 - alpha_mid) * predicted_noise

            # Predict noise at midpoint
            predicted_noise_mid = self._get_predicted_noise(model, x_mid, t_mid, labels, cfg_scale)

            # Step 4: Update x using midpoint correction
            x = torch.sqrt(alpha_prev) * x0 + torch.sqrt(1 - alpha_prev) * predicted_noise_mid

            # Add noise if using stochastic sampling (eta > 0)
            if self.eta > 0 and i > 1:
                sigma = self.eta * torch.sqrt(
                    (1 - alpha_prev) / (1 - alpha_curr) * (1 - alpha_curr / alpha_prev)
                )
                x += sigma * torch.randn_like(x)

        return x
