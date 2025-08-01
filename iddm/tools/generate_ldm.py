#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Date   : 2025/8/1 11:08
    @Author : chairc
    @Site   : https://github.com/chairc
"""
import os
import sys
import argparse
import time

import torch
import logging
import coloredlogs

from iddm.model.samples.ldm import LDMDiffusion

sys.path.append(os.path.dirname(sys.path[0]))
from iddm.config.choices import ldm_sample_choices, image_format_choices, \
    parse_image_size_type
from iddm.config.version import get_version_banner
from iddm.utils.check import check_image_size
from iddm.utils.initializer import device_initializer, network_initializer, generate_initializer, \
    autoencoder_network_initializer, generate_autoencoder_initializer
from iddm.utils.utils import plot_images, save_images, save_one_image_in_images, check_and_create_dir
from iddm.utils.checkpoint import load_ckpt

logger = logging.getLogger(__name__)
coloredlogs.install(level="INFO")


class Generator:
    """
    Diffusion model generator
    """

    def __init__(self, gen_args, deploy=False):
        """
        Generating init
        :param gen_args: Input parameters
        :param deploy: App deploy
        :return: None
        """
        self.args = gen_args
        self.deploy = deploy

        logger.info(msg="Start generation.")
        logger.info(msg=f"Input params: {self.args}")
        # Diffusion weight path
        self.diffusion_path = self.args.diffusion_path
        # Autoencoder weight path
        self.autoencoder_path = self.args.autoencoder_path
        # Run device initializer
        self.device = device_initializer(device_id=self.args.use_gpu)
        # Enable conditional generation, sample type, network, image size,
        # number of classes and select activation function
        gen_results = generate_initializer(ckpt_path=self.diffusion_path, conditional=None, image_size=None,
                                           sample=None, network=None, act=None, num_classes=None, device=self.device)
        gen_autoencoder_results = generate_autoencoder_initializer(ckpt_path=self.autoencoder_path, device=self.device)
        self.conditional, self.network, self.image_size, self.num_classes, self.act = gen_results
        (self.autoencoder_network, self.autoencoder_image_size, self.autoencoder_latent_channels,
         self.autoencoder_act) = gen_autoencoder_results
        # Check image size format
        self.image_size = check_image_size(image_size=self.image_size)
        # Generation name
        self.generate_name = self.args.generate_name
        # Sample
        self.sample = self.args.sample
        # Number of images
        self.num_images = self.args.num_images
        # Use ema
        self.use_ema = self.args.use_ema
        # Format of images
        self.image_format = self.args.image_format
        # Saving path
        self.result_path = os.path.join(self.args.result_path, str(time.time()))
        # Check and create result path
        if not deploy:
            check_and_create_dir(self.result_path)
        # Init diffusion network
        self.Network = network_initializer(network=self.network, device=self.device)
        # Init autoencoder network
        autoencoder_network = autoencoder_network_initializer(network=self.autoencoder_network, device=self.device)
        self.autoencoder = autoencoder_network(latent_channels=self.autoencoder_latent_channels,
                                               device=self.device).to(self.device)
        load_ckpt(ckpt_path=self.autoencoder_path, model=self.autoencoder, is_train=False, device=self.device)
        # Inference mode, no updating parameters
        self.autoencoder.eval()
        # Initialize the diffusion model
        # If you want to ignore the rules and generate a large image, modify image_size=[h,w]
        self.diffusion = LDMDiffusion(autoencoder=self.autoencoder, img_size=self.image_size, device=self.device)
        # Is it necessary to expand the image?
        self.input_image_size = check_image_size(image_size=self.args.image_size)
        # Initialize model
        if self.conditional:
            # Generation class name
            self.class_name = self.args.class_name
            # classifier-free guidance interpolation weight
            self.cfg_scale = self.args.cfg_scale
            # If you want to ignore the rules and generate a large image, modify image_size=[h,w]
            self.model = self.Network(in_channel=self.autoencoder_latent_channels,
                                      out_channel=self.autoencoder_latent_channels, num_classes=self.num_classes,
                                      device=self.device, image_size=self.image_size,
                                      act=self.act).to(self.device)
            load_ckpt(ckpt_path=self.diffusion_path, model=self.model, device=self.device, is_train=False,
                      is_use_ema=self.use_ema, conditional=self.conditional)
        else:
            # If you want to ignore the rules and generate a large image, modify image_size=[h,w]
            self.model = self.Network(in_channel=self.autoencoder_latent_channels,
                                      out_channel=self.autoencoder_latent_channels, device=self.device,
                                      image_size=self.image_size, act=self.act).to(self.device)
            load_ckpt(ckpt_path=self.diffusion_path, model=self.model, device=self.device, is_train=False,
                      conditional=self.conditional)

    def generate(self, index=0):
        """
        Generate images
        :param index: Image index
        """
        if self.conditional:
            if self.class_name == -1:
                y = torch.arange(self.num_classes).long().to(self.device)
                self.num_images = self.num_classes
            else:
                y = torch.Tensor([self.class_name] * self.num_images).long().to(self.device)
            x = self.diffusion.sample(model=self.model, n=self.num_images, labels=y, cfg_scale=self.cfg_scale)
        else:
            x = self.diffusion.sample(model=self.model, n=self.num_images)

        # If deploy app is true, return the generate results
        if self.deploy:
            return x

        if self.result_path == "" or self.result_path is None:
            plot_images(images=x)
        else:
            save_name = f"{self.generate_name}_{index}"
            save_images(images=x, path=os.path.join(self.result_path, f"{save_name}.{self.image_format}"))
            save_one_image_in_images(images=x, path=self.result_path, generate_name=save_name,
                                     image_size=self.image_size, image_format=self.image_format)
            plot_images(images=x)
        logger.info(msg="Finish generation.")


def init_generate_args():
    # Generating model parameters
    # required: Must be set
    # needed: Set as needed
    # recommend: Recommend to set
    parser = argparse.ArgumentParser()
    # =================================Base settings=================================
    # Generation name (required)
    parser.add_argument("--generate_name", "-n", type=str, default="ldm")
    # Number of generation images (required)
    # if class name is `-1` and conditional `is` True, the model would output one image per class.
    parser.add_argument("--num_images", type=int, default=1)
    # Use ema model
    # If set to false, the pt file of the ordinary model will be used
    # If true, the pt file of the ema model will be used
    parser.add_argument("--use_ema", default=False, action="store_true")
    # Weight path (required)
    parser.add_argument("--autoencoder_path", type=str, default="/your/path/of/autoencoder/model.pt")
    # Weight path (required)
    parser.add_argument("--diffusion_path", type=str, default="/your/path/of/diffusion/model.pt")
    # Saving path (required)
    parser.add_argument("--result_path", type=str, default="/your/path/Defect-Diffusion-Model/results/vis")
    # Generated image format
    # Recommend to use png for better generation quality.
    # Option: jpg/png
    parser.add_argument("--image_format", type=str, default="png", choices=image_format_choices)
    # Set the sample type (required)
    # Option: ldm
    parser.add_argument("--sample", type=str, default="ldm", choices=ldm_sample_choices)
    # Diffusion image size (required)
    parser.add_argument("--image_size", "-i", type=parse_image_size_type, default=64)

    # =====================Enable the conditional generation (if '--conditional' is set to 'True')=====================
    # Class name (required)
    # if class name is `-1`, the model would output one image per class.
    # [Note] The setting range should be [0, num_classes - 1].
    parser.add_argument("--class_name", type=int, default=0)
    # classifier-free guidance interpolation weight, users can better generate model effect (recommend)
    parser.add_argument("--cfg_scale", type=int, default=3)
    # Set the use GPU in generate (required)
    parser.add_argument("--use_gpu", type=int, default=0)

    return parser.parse_args()


if __name__ == "__main__":
    # Init generate args
    args = init_generate_args()
    # Get version banner
    get_version_banner()
    gen_model = Generator(gen_args=args, deploy=False)
    for i in range(2):
        gen_model.generate(index=i)
