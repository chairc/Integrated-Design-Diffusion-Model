#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Date   : 2023/7/21 23:06
    @Author : chairc
    @Site   : https://github.com/chairc
"""
import math
from typing import Optional, List, Union, Tuple

import torch
from torch import nn

from iddm.config.setting import DEFAULT_IMAGE_SIZE, IMAGE_CHANNEL


class BaseDiffusion:
    """
    Base diffusion class
    """

    def __init__(
            self,
            noise_steps: int = 1000,
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
        Diffusion model base class
        :param noise_steps: Noise steps
        :param beta_start: β start
        :param beta_end: β end
        :param img_size: Image size
        :param device: Device type
        :param schedule_name: Prepare the noise schedule name
        :param latent: Whether to use latent diffusion
        :param latent_channel: Latent channel size, default is 8
        :param autoencoder: Autoencoder, used for latent diffusion
        """
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = None
        self.device = device
        self.schedule_name = schedule_name
        # Image channel, RGB or GRAY
        self.image_channel = IMAGE_CHANNEL

        # Init image size
        self._init_sample_image_size(img_size=img_size)

        # Latent diffusion
        # Whether to use latent diffusion
        self.latent = latent
        if self.latent:
            # Latent channel size
            self.latent_channel = latent_channel
            # Autoencoder, used for latent diffusion
            self.autoencoder = autoencoder
            self._init_autoencoder()

        # Noise steps
        self.beta = self.prepare_noise_schedule(schedule_name=self.schedule_name).to(self.device)
        # Formula: α = 1 - β
        self.alpha = 1. - self.beta
        # The cumulative sum of α.
        self.alpha_hat = torch.cumprod(input=self.alpha, dim=0)

    def prepare_noise_schedule(
            self,
            schedule_name: str = "linear"
    ) -> torch.Tensor:
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

    def noise_images(
            self,
            x: torch.Tensor,
            time: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
        eps = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * eps, eps

    def sample_time_steps(
            self,
            n: int
    ) -> torch.Tensor:
        """
        Sample time steps
        :param n: Image size
        :return: Integer tensor of shape (n,)
        """
        # Generate a tensor of integers with the specified shape (n,)
        # where each element is randomly chosen between low and high (contains low, does not contain high)
        return torch.randint(low=1, high=self.noise_steps, size=(n,), device=self.device)

    def _init_sample_image_size(
            self,
            img_size: Optional[List[int]] = None
    ):
        """
        Initialize sample image size
        :param img_size: Image size
        :return: Integer tensor of shape [image_size_h, image_size_w]
        """
        if img_size is None:
            self.img_size = DEFAULT_IMAGE_SIZE
        else:
            self.img_size = img_size

    def sample(
            self,
            model: nn.Module,
            x: Optional[torch.Tensor] = None,
            n: int = 1,
            labels: Optional[torch.Tensor] = None,
            cfg_scale: Optional[float] = None
    ) -> torch.Tensor:
        """
        Sample method, this method should be implemented in the subclass
        :param model: Model
        :param x: Input image tensor, if provided, will be used as the starting point for sampling
        :param n: Number of sample images, x priority is greater than n
        :param labels: Labels
        :param cfg_scale: classifier-free guidance interpolation weight, users can better generate model effect.
        :return: Sample images
        """
        pass

    def _get_input_image(
            self,
            n: int,
            x: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, int]:
        """
        Get input image tensor
        :param x: Input image tensor, if provided, will be used as the starting point for sampling
        :return: Input image tensor
        """
        if self.latent:
            channel = self.latent_channel
        else:
            channel = self.image_channel
        if x is None and n > 0:
            # If no input image is provided, generate a random noise image
            return torch.randn((n, channel, self.img_size[0], self.img_size[1])).to(self.device), n
        elif x is not None:
            # If an input image is provided, ensure it has the correct shape
            return x.to(self.device), x.shape[0]
        else:
            # If no input image is provided and n is 0, return a random noise image with batch size 1
            return torch.randn((1, channel, self.img_size[0], self.img_size[1])).to(self.device), 1

    @staticmethod
    def _get_predicted_noise(
            model: nn.Module,
            x: torch.Tensor,
            t: torch.Tensor,
            labels: Optional[torch.Tensor],
            cfg_scale: Optional[float]
    ) -> torch.Tensor:
        """
        Obtaining Predictive Noise (Public Logic Extraction)
        Handle conditional generation and classifier guidance
        :param model: Noise prediction model
        :param x: Input image tensor
        :param t: Time step tensor
        :param labels: Conditional labels (optional)
        :param cfg_scale: Classifier-free guidance scale (optional)
        :return: Predicted noise tensor
        """
        # Whether the network has conditional input, such as multiple category input
        if labels is None or cfg_scale is None or cfg_scale <= 0:
            # Images and time steps input into the model
            return model(x, t) if labels is None else model(x, t, labels)

        # Classifier guidance: mixed conditional and unconditional predictions
        conditional_noise = model(x, t, labels)
        # Avoiding the posterior collapse problem and better generate model effect
        # Unconditional predictive noise
        unconditional_noise = model(x, t, None)
        # 'torch.lerp' performs linear interpolation between the start and end values
        # according to the given weights
        # Formula: input + weight * (end - input)
        return torch.lerp(unconditional_noise, conditional_noise, cfg_scale)

    def post_process(
            self,
            x: torch.Tensor
    ) -> torch.Tensor:
        """
        Post process
        :param x: Input image tensor must range of -1 and 1
        :return: Post process tensor
        """
        # Latent mode
        if self.latent:
            # If latent diffusion, the output is a tensor of shape [n, latent_channel, img_size_h, img_size_w]
            # The latent channel size is 8, so we need to convert it to 3 channels
            x = self.autoencoder.decode(x)
        # Return the value to the range of 0 and 1
        x = (x + 1) * 0.5
        # Multiply by 255 to enter the effective pixel range
        x = (x * 255).type(torch.uint8)
        return x

    def _init_autoencoder(self):
        """
        Initialize the autoencoder and check its type
        :return: None
        """
        if self.autoencoder is not None:
            if not isinstance(self.autoencoder, nn.Module):
                raise TypeError("'autoencoder' must be an instance of nn.Module")
            self.autoencoder = self.autoencoder.to(self.device)
