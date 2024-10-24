#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Date   : 2023/7/21 23:06
    @Author : chairc
    @Site   : https://github.com/chairc
"""
import math

import torch

from config.setting import DEFAULT_IMAGE_SIZE


class BaseDiffusion:
    """
    Base diffusion class
    """

    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=2e-2, img_size=None, device="cpu",
                 schedule_name="linear"):
        """
        Diffusion model base class
        :param noise_steps: Noise steps
        :param beta_start: β start
        :param beta_end: β end
        :param img_size: Image size
        :param device: Device type
        :param schedule_name: Prepare the noise schedule name
        """
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = None
        self.device = device
        self.schedule_name = schedule_name

        # Init image size
        self.init_sample_image_size(img_size=img_size)

        # Noise steps
        self.beta = self.prepare_noise_schedule(schedule_name=self.schedule_name).to(self.device)
        # Formula: α = 1 - β
        self.alpha = 1. - self.beta
        # The cumulative sum of α.
        self.alpha_hat = torch.cumprod(input=self.alpha, dim=0)

    def prepare_noise_schedule(self, schedule_name="linear"):
        """
        Prepare the noise schedule
        :param schedule_name: Function, linear and cosine
        :return: schedule
        """
        if schedule_name == "linear":
            # 'torch.linspace' generates a 1-dimensional tensor for the specified interval,
            # and the values in it are evenly distributed
            return torch.linspace(start=self.beta_start, end=self.beta_end, steps=self.noise_steps)
        elif schedule_name == "cosine":
            def alpha_hat(t):
                """
                The parameter t ranges from 0 to 1
                Generate (1 - β) to the cumulative product of this part of the diffusion process
                The original formula â is calculated as: α_hat(t) = f(t) / f(0)
                The original formula f(t) is calculated as: f(t) = cos(((t / (T + s)) / (1 + s)) · (π / 2))²
                In this function, s = 0.008 and f(0) = 1
                So just return f(t)
                :param t: Time
                :return: The value of alpha_hat at t
                """
                return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2

            # Number of betas to generate
            noise_steps = self.noise_steps
            # The max value of β, use a value less than 1 to prevent singularities
            max_beta = 0.999
            # Create a beta schedule that scatter given the alpha_hat(t) function,
            # defining the cumulative product of (1 - β) from t = [0,1]
            betas = []
            # Loop
            for i in range(noise_steps):
                t1 = i / noise_steps
                t2 = (i + 1) / noise_steps
                # Calculate the value of β at time t
                # Formula: β(t) = min(1 - (α_hat(t) - α_hat(t-1)), 0.999)
                beta_t = min(1 - alpha_hat(t2) / alpha_hat(t1), max_beta)
                betas.append(beta_t)
            return torch.tensor(betas)
        elif schedule_name == "sqrt_linear":
            return torch.linspace(start=self.beta_start ** 0.5, end=self.beta_end ** 0.5, steps=self.noise_steps) ** 2
        elif schedule_name == "sqrt":
            return torch.linspace(start=self.beta_start, end=self.beta_end, steps=self.noise_steps) ** 0.5
        else:
            raise NotImplementedError(f"Unknown beta schedule: {schedule_name}")

    def noise_images(self, x, time):
        """
        Add noise to the image
        :param x: Input image
        :param time: Time
        :return: sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, a tensor of the same shape as the x tensor at time t
        """
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[time])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[time])[:, None, None, None]
        # Generates a tensor of the same shape as the x tensor,
        # with elements randomly sampled from a standard normal distribution (mean 0, variance 1)
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

    def sample_time_steps(self, n):
        """
        Sample time steps
        :param n: Image size
        :return: Integer tensor of shape (n,)
        """
        # Generate a tensor of integers with the specified shape (n,)
        # where each element is randomly chosen between low and high (contains low, does not contain high)
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def init_sample_image_size(self, img_size):
        """
        Initialize sample image size
        :param img_size: Image size
        :return: Integer tensor of shape [image_size_h, image_size_w]
        """
        if img_size is None:
            self.img_size = DEFAULT_IMAGE_SIZE
        else:
            self.img_size = img_size
