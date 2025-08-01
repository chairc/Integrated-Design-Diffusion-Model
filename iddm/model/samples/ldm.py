#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Date   : 2025/07/28 14:47
    @Author : chairc
    @Site   : https://github.com/chairc
"""
from typing import Optional, List, Union

import torch
import logging
import coloredlogs
from torch import nn
from tqdm import tqdm

from iddm.config.setting import LATENT_CHANNEL
from iddm.model.samples.ddim import DDIMDiffusion

logger = logging.getLogger(__name__)
coloredlogs.install(level="INFO")


class LDMDiffusion(DDIMDiffusion):
    """
    Latent Diffusion class
    Based on the latent space diffusion model,
    the image is mapped to the low-dimensional latent space through the autoencoder for the diffusion process
    """

    def __init__(
            self,
            autoencoder: nn.Module,
            noise_steps: int = 1000,
            sample_steps: int = 100,
            beta_start: float = 1e-4,
            beta_end: float = 2e-2,
            latent_size: int = LATENT_CHANNEL,
            img_size: Optional[List[int]] = None,
            autoencoder_ckpt: str = None,
            device: Union[str, torch.device] = "cpu",
            schedule_name: str = "linear",
    ):
        """
        The implement of LDM
        :param autoencoder: Pre-trained autoencoder model, if None, will use the default Autoencoder class
        :param noise_steps: Noise steps
        :param sample_steps: Sample steps
        :param beta_start: β start
        :param beta_end: β end
        :param latent_size: Latent space size, if None, will be set to img_size // 8 (default downsample 8 times)
        :param img_size: Image size, it is diffusion model input image size
        :param autoencoder_ckpt: Path to the pre-trained autoencoder checkpoint
        :param device: Device type
        :param schedule_name: Prepare the noise schedule name
        """
        # Call base class initialization
        super().__init__(noise_steps, sample_steps, beta_start, beta_end, img_size, device, schedule_name)

        # Subspatial configuration
        self.latent_size = latent_size or (img_size[0] // 8, img_size[1] // 8)
        self.image_channel = self.latent_size

        # Initializing the autoencoder (for image <-> latent space conversion)
        if autoencoder is not None:
            if not isinstance(autoencoder, nn.Module):
                raise TypeError("'autoencoder' must be an instance of nn.Module")
            self.autoencoder = autoencoder.to(device)
        else:
            raise ValueError("'autoencoder' must be provided as an instance of nn.Module")

        # Load the pre-trained autoencoder weights
        if autoencoder_ckpt:
            self._load_autoencoder(ckpt_path=autoencoder_ckpt)

    def _load_autoencoder(self, ckpt_path):
        """
        Load the pre-trained autoencoder weights
        :param ckpt_path: Path to the autoencoder checkpoint
        :return: None
        """
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        self.autoencoder.load_state_dict(ckpt["model"])
        self.autoencoder.eval()
        logger.info(f"Loaded autoencoder from {ckpt_path}")

    def encode(self, x):
        """
        Encode the image into the subconscious space
        :param x: Input image in the shape [B, C, H, W]
        :return: latent variable in the form of [B, 4, latent_size[0], latent_size[1]]
        """
        with torch.no_grad():
            z = self.autoencoder.encode(x)
            # Standardized latent variables (similar to Stable Diffusion)
            # Normalized coefficients for pre-trained autoencoders
            z = z * 0.18215
            return z

    def decode(self, z):
        """
        Decode latent variables into images
        :param z: latent variable in the form [B, 4, latent_size[0], latent_size[1]]
        :return: The decoded image in the shape [B, C, H, W]
        """
        with torch.no_grad():
            # Reverse normalization
            z = z / 0.18215
            x = self.autoencoder.decode(z)
            return x

    def sample(
            self,
            model: nn.Module,
            x: Optional[torch.Tensor] = None,
            n: int = 1,
            labels: Optional[torch.Tensor] = None,
            cfg_scale: Optional[float] = None
    ) -> torch.Tensor:
        """
        Latent diffusion sampling process
        1. Generate random noise in subconscious space
        2. Diffusion model denoising to obtain latent variables
        3. Decode the latent variables to get the final image
        :param model: Model
        :param x: Input image tensor, if provided, will be used as the starting point for sampling
        :param n: Number of sample images, x priority is greater than n
        :param labels: Labels
        :param cfg_scale: classifier-free guidance interpolation weight, users can better generate model effect.
        Avoiding the posterior collapse problem, Reference paper: 'Classifier-Free Diffusion Guidance'
        :return: Sample images
        """
        # TODO: The use of various samplers is supported in the future
        # Step1: Initialize latent space noise (number of channels is autoencoder output channel)
        z = torch.randn((n, self.latent_size, self.img_size[0], self.img_size[1])).to(self.device)
        logger.info(f"Latent Diffusion Sampling {n} new images....")
        model.eval()
        with torch.no_grad():
            # Step2: Latent space diffusion denoising
            for i, p_i in tqdm(self.time_step):
                t = (torch.ones(n) * i).long().to(self.device)
                p_t = (torch.ones(n) * p_i).long().to(self.device)

                alpha_t = self.alpha_hat[t][:, None, None, None]
                alpha_prev = self.alpha_hat[p_t][:, None, None, None]

                noise = torch.randn_like(z) if i > 1 else torch.zeros_like(z)

                # Predict noise (diffusion model input as a latent variable)
                predicted_noise = self._get_predicted_noise(model, z, t, labels, cfg_scale)

                # Calculation formula, latent space denoising uses DDIM style
                # Division would cause the value to be too large or too small, and it needs to be constrained
                # https://github.com/ermongroup/ddim/blob/main/functions/denoising.py#L54C12-L54C54
                z0_t = torch.clamp((z - predicted_noise * torch.sqrt(1 - alpha_t)) / torch.sqrt(alpha_t), -1, 1)
                # Sigma
                c1 = self.eta * torch.sqrt((1 - alpha_t / alpha_prev) * (1 - alpha_prev) / (1 - alpha_t))
                c2 = torch.sqrt((1 - alpha_prev) - c1 ** 2)
                # Predicted x0 + direction pointing to xt + sigma * predicted noise
                z = torch.sqrt(alpha_prev) * z0_t + c2 * predicted_noise + c1 * noise

            # Step3: Decode the latent variables to get the image
            x = self.decode(z)

        model.train()
        x = self.post_process(x=x)
        return x
