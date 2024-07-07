#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Date   : 2024/2/6 3:19
    @Author : chairc
    @Site   : https://github.com/chairc
"""
import torch
import logging
import coloredlogs

from tqdm import tqdm

from model.samples.base import BaseDiffusion

logger = logging.getLogger(__name__)
coloredlogs.install(level="INFO")


class PLMSDiffusion(BaseDiffusion):
    """
    PLMS class
    """

    def __init__(self, noise_steps=1000, sample_steps=100, beta_start=1e-4, beta_end=2e-2, img_size=None, device="cpu",
                 schedule_name="linear"):
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
        super().__init__(noise_steps, beta_start, beta_end, img_size, device, schedule_name)
        # Sample steps, it skips some steps
        self.sample_steps = sample_steps

        self.eta = 0

        # Calculate time step size, it skips some steps
        self.time_step = torch.arange(0, self.noise_steps, (self.noise_steps // self.sample_steps)).long() + 1
        self.time_step = reversed(torch.cat((torch.tensor([0], dtype=torch.long), self.time_step)))
        self.time_step = list(zip(self.time_step[:-1], self.time_step[1:]))

    def sample(self, model, n, labels=None, cfg_scale=None):
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
            # Input dim: [n, 3, img_size_h, img_size_w]
            x = torch.randn((n, 3, self.img_size[0], self.img_size[1])).to(self.device)
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
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                # Whether the network has conditional input, such as multiple category input
                if labels is None and cfg_scale is None:
                    # Images and time steps input into the model
                    predicted_noise = model(x, t)
                else:
                    predicted_noise = model(x, t, labels)
                    # Avoiding the posterior collapse problem and better generate model effect
                    if cfg_scale > 0:
                        # Unconditional predictive noise
                        unconditional_predicted_noise = model(x, t, None)
                        # 'torch.lerp' performs linear interpolation between the start and end values
                        # according to the given weights
                        # Formula: input + weight * (end - input)
                        predicted_noise = torch.lerp(unconditional_predicted_noise, predicted_noise, cfg_scale)
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
        model.train()
        # Return the value to the range of 0 and 1
        x = (x + 1) * 0.5
        # Multiply by 255 to enter the effective pixel range
        x = (x * 255).type(torch.uint8)
        return x
