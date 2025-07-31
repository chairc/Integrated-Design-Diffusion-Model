#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Date   : 2023/6/15 17:12
    @Author : chairc
    @Site   : https://github.com/chairc
"""
from typing import Optional, List, Union

import torch
import logging
import coloredlogs
from torch import nn

from tqdm import tqdm

from iddm.model.samples.base import BaseDiffusion

logger = logging.getLogger(__name__)
coloredlogs.install(level="INFO")


class DDPMDiffusion(BaseDiffusion):
    """
    DDPM class
    """

    def __init__(
            self,
            noise_steps: int = 1000,
            beta_start: float = 1e-4,
            beta_end: float = 2e-2,
            img_size: Optional[List[int]] = None,
            device: Union[str, torch.device] = "cpu",
            schedule_name: str = "linear"
    ):
        """
        The implement of DDPM
        Paper: Denoising Diffusion Probabilistic Models
        URL: https://arxiv.org/abs/2006.11239
        :param noise_steps: Noise steps
        :param beta_start: β start
        :param beta_end: β end
        :param img_size: Image size
        :param device: Device type
        :param schedule_name: Prepare the noise schedule name
        """

        super().__init__(noise_steps, beta_start, beta_end, img_size, device, schedule_name)

    def sample(
            self,
            model: nn.Module,
            x: Optional[torch.Tensor] = None,
            n: int = 1,
            labels: Optional[torch.Tensor] = None,
            cfg_scale: Optional[float] = None
    ) -> torch.Tensor:
        """
        DDPM sample method
        :param model: Model
        :param x: Input image tensor, if provided, will be used as the starting point for sampling
        :param n: Number of sample images, x priority is greater than n
        :param labels: Labels
        :param cfg_scale: classifier-free guidance interpolation weight, users can better generate model effect.
        Avoiding the posterior collapse problem, Reference paper: 'Classifier-Free Diffusion Guidance'
        :return: Sample images
        """
        # Input dim: [n, img_channel, img_size_h, img_size_w]
        x, n = self._get_input_image(n=n, x=x)
        logger.info(msg=f"DDPM Sampling {n} new images....")
        model.eval()
        with torch.no_grad():
            # 'reversed(range(1, self.noise_steps)' iterates over a sequence of integers in reverse
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0, total=self.noise_steps - 1):
                # Time step, creating a tensor of size n
                t = (torch.ones(n) * i).long().to(self.device)

                # Whether the network has conditional input, such as multiple category input
                # Predict noise
                predicted_noise = self._get_predicted_noise(model, x, t, labels, cfg_scale)

                # Expand to a 4-dimensional tensor, and get the value according to the time step t
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                # Only noise with a step size greater than 1 is required.
                # For details, refer to line 3 of Algorithm 2 on page 4 of the paper
                noise = torch.randn_like(x) if i > 1 else torch.zeros_like(x)

                # In each epoch, use x to calculate t - 1 of x
                # For details, refer to line 4 of Algorithm 2 on page 4 of the paper
                x = 1 / torch.sqrt(alpha) * (
                        x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(
                    beta) * noise
        model.train()
        # Post process
        x = self.post_process(x=x.clamp(-1, 1))
        return x
