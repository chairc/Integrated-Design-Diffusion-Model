#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Date   : 2024/2/6 3:19
    @Author : chairc
    @Site   : https://github.com/chairc
"""
from typing import Optional, List, Union

import torch
import logging
import coloredlogs
from torch import nn

from tqdm import tqdm

from iddm.model.samples.ddim import DDIMDiffusion

logger = logging.getLogger(__name__)
coloredlogs.install(level="INFO")


class PLMSDiffusion(DDIMDiffusion):
    """
    PLMS class
    """

    def __init__(
            self,
            noise_steps: int = 1000,
            sample_steps: int = 100,
            beta_start: float = 1e-4,
            beta_end: float = 2e-2,
            img_size: Optional[List[int]] = None,
            device: Union[str, torch.device] = "cpu",
            schedule_name: str = "linear"
    ):
        """
        The implement of PLMS, like DDIM
        Paper: Pseudo Numerical Methods for Diffusion Models on Manifolds
        URL: https://openreview.net/forum?id=PlKWVd2yBkY
        :param noise_steps: Noise steps
        :param sample_steps: Sample steps
        :param beta_start: β start
        :param beta_end: β end
        :param img_size: Image size
        :param device: Device type
        :param schedule_name: Prepare the noise schedule name
        """
        super().__init__(noise_steps, sample_steps, beta_start, beta_end, img_size, device, schedule_name)

    def sample(
            self,
            model: nn.Module,
            n: int,
            labels: Optional[torch.Tensor] = None,
            cfg_scale: Optional[float] = None
    ) -> torch.Tensor:
        """
        PLMS sample method
        :param model: Model
        :param n: Number of sample images
        :param labels: Labels
        :param cfg_scale: classifier-free guidance interpolation weight, users can better generate model effect.
        Avoiding the posterior collapse problem, Reference paper: 'Classifier-Free Diffusion Guidance'
        :return: Sample images
        """
        logger.info(msg=f"PLMS Sampling {n} new images....")
        model.eval()
        with torch.no_grad():
            # Input dim: [n, img_channel, img_size_h, img_size_w]
            x = torch.randn((n, self.image_channel, self.img_size[0], self.img_size[1])).to(self.device)
            # Old eps
            old_eps = []
            # The list of current time and previous time
            for i, p_i in tqdm(self.time_step):
                # Time step, creating a tensor of size n
                t = (torch.ones(n) * i).long().to(self.device)
                # Previous time step, creating a tensor of size n
                p_t = (torch.ones(n) * p_i).long().to(self.device)
                # Expand to a 4-dimensional tensor, and get the value according to the time step t
                alpha_t = self.alpha_hat[t][:, None, None, None]
                alpha_prev = self.alpha_hat[p_t][:, None, None, None]
                noise = torch.randn_like(x) if i > 1 else torch.zeros_like(x)
                # Predict noise
                predicted_noise = self._get_predicted_noise(model, x, t, labels, cfg_scale)
                # Calculation formula
                if len(old_eps) == 0:
                    # Pseudo Improved Euler (2nd order)
                    x0_t = torch.clamp((x - (predicted_noise * torch.sqrt((1 - alpha_t)))) / torch.sqrt(alpha_t), -1, 1)
                    c1 = self.eta * torch.sqrt((1 - alpha_t / alpha_prev) * (1 - alpha_prev) / (1 - alpha_t))
                    c2 = torch.sqrt((1 - alpha_prev) - c1 ** 2)
                    p_x = torch.sqrt(alpha_prev) * x0_t + c2 * predicted_noise + c1 * noise
                    if labels is None and cfg_scale is None:
                        # Images and time steps input into the model
                        predicted_noise_next = model(p_x, p_t)
                    else:
                        predicted_noise_next = model(p_x, p_t, labels)
                    predicted_noise_prime = (predicted_noise + predicted_noise_next) / 2
                elif len(old_eps) == 1:
                    # 2nd order Pseudo Linear Multistep (Adams-Bashforth)
                    predicted_noise_prime = (3 * predicted_noise - old_eps[-1]) / 2
                elif len(old_eps) == 2:
                    # 3rd order Pseudo Linear Multistep (Adams-Bashforth)
                    predicted_noise_prime = (23 * predicted_noise - 16 * old_eps[-1] + 5 * old_eps[-2]) / 12
                elif len(old_eps) >= 3:
                    # 4th order Pseudo Linear Multistep (Adams-Bashforth)
                    predicted_noise_prime = (55 * predicted_noise - 59 * old_eps[-1] + 37 * old_eps[-2] -
                                             9 * old_eps[-3]) / 24

                x0_t = torch.clamp((x - (predicted_noise_prime * torch.sqrt((1 - alpha_t)))) / torch.sqrt(alpha_t),
                                   -1, 1)
                c1 = self.eta * torch.sqrt((1 - alpha_t / alpha_prev) * (1 - alpha_prev) / (1 - alpha_t))
                c2 = torch.sqrt((1 - alpha_prev) - c1 ** 2)
                x = torch.sqrt(alpha_prev) * x0_t + c2 * predicted_noise_prime + c1 * noise
                # Save old predicted_noise
                old_eps.append(predicted_noise)
                # Only the last 3 historical values are retained to save memory
                if len(old_eps) > 3:
                    old_eps.pop(0)
        model.train()
        # Post process
        x = self.post_process(x=x)
        return x
