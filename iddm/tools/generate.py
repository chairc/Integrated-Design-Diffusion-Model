#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Date   : 2023/6/20 22:33
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

sys.path.append(os.path.dirname(sys.path[0]))
from iddm.config import IMAGE_CHANNEL
from iddm.config.choices import sample_choices, image_format_choices, parse_image_size_type, network_choices, \
    act_choices, generate_mode_choices
from iddm.config.version import get_version_banner
from iddm.utils.check import check_image_size
from iddm.utils.initializer import device_initializer, network_initializer, sample_initializer, generate_initializer, \
    generate_autoencoder_initializer, autoencoder_network_initializer
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
        # Init parameters
        self.in_channels, self.out_channels = IMAGE_CHANNEL, IMAGE_CHANNEL
        (self.autoencoder, self.autoencoder_network, self.autoencoder_image_size, self.autoencoder_latent_channels,
         self.autoencoder_act) = None, None, None, None, None
        self.input_image, self.input_text = None, None

        # Latent diffusion model
        self.latent = self.args.latent
        # Weight path
        self.weight_path = self.args.weight_path
        # Autoencoder weight path
        self.autoencoder_ckpt = self.args.autoencoder_ckpt
        # Run device initializer
        self.device = device_initializer(device_id=self.args.use_gpu)

        # Enable conditional generation, sample type, network, image size,
        # number of classes and select activation function
        gen_results = generate_initializer(ckpt_path=self.weight_path, conditional=self.args.conditional,
                                           image_size=self.args.image_size, sample=self.args.sample,
                                           network=self.args.network, act=self.args.act,
                                           num_classes=self.args.num_classes, device=self.device)
        self.conditional, self.network, self.image_size, self.num_classes, self.act = gen_results

        # Check image size format
        self.image_size = check_image_size(image_size=self.image_size)
        self.resize_image_size = self.image_size
        # Sample
        self.sample = self.args.sample
        # Generation name
        self.generate_name = f"{self.args.generate_name}_{self.sample}"
        # Number of images
        self.num_images = self.args.num_images
        # Use ema
        self.use_ema = self.args.use_ema
        # Format of images
        self.image_format = self.args.image_format
        # Saving path
        self.result_path = os.path.join(self.args.result_path, str(time.time()))
        # Conditional generation mode check
        self.mode = self.args.mode
        # Check and create result path
        if not deploy:
            check_and_create_dir(self.result_path)
        # Network
        self.Network = network_initializer(network=self.network, device=self.device)

        # If latent diffusion model is enabled, get autoencoder results
        if self.latent:
            gen_autoencoder_results = generate_autoencoder_initializer(ckpt_path=self.autoencoder_ckpt,
                                                                       device=self.device)
            (self.autoencoder_network, self.autoencoder_image_size, self.autoencoder_latent_channels,
             self.autoencoder_act) = gen_autoencoder_results
            self.in_channels, self.out_channels = self.autoencoder_latent_channels, self.autoencoder_latent_channels
            self.resize_image_size = check_image_size(image_size=self.autoencoder_image_size)

            # Init autoencoder network
            autoencoder_network = autoencoder_network_initializer(network=self.autoencoder_network, device=self.device)
            self.autoencoder = autoencoder_network(latent_channels=self.autoencoder_latent_channels,
                                                   device=self.device).to(self.device)
            load_ckpt(ckpt_path=self.autoencoder_ckpt, model=self.autoencoder, is_generate=True, device=self.device)
            # Inference mode, no updating parameters
            self.autoencoder.eval()

        # Initialize the diffusion model
        self.diffusion = sample_initializer(sample=self.sample, image_size=self.image_size, device=self.device,
                                            latent=self.latent, latent_channel=self.autoencoder_latent_channels,
                                            autoencoder=self.autoencoder)

        # Is it necessary to expand the image?
        self.resize_image_size = check_image_size(image_size=self.resize_image_size)
        if self.image_size == self.resize_image_size:
            self.new_image_size = None
        else:
            self.new_image_size = self.resize_image_size
        # Initialize model
        if self.conditional:
            # Generation class name
            self.class_name = self.args.class_name
            # classifier-free guidance interpolation weight
            self.cfg_scale = self.args.cfg_scale
            # If you want to ignore the rules and generate a large image, modify image_size=[h,w]
            self.model = self.Network(mode=self.mode, in_channel=self.in_channels, out_channel=self.out_channels,
                                      num_classes=self.num_classes, device=self.device, image_size=self.image_size,
                                      act=self.act).to(self.device)
            load_ckpt(ckpt_path=self.weight_path, model=self.model, device=self.device, is_generate=True,
                      is_use_ema=self.use_ema, conditional=self.conditional)
        else:
            # If you want to ignore the rules and generate a large image, modify image_size=[h,w]
            self.model = self.Network(device=self.device, image_size=self.image_size, act=self.act).to(self.device)
            load_ckpt(ckpt_path=self.weight_path, model=self.model, device=self.device, is_generate=True,
                      conditional=self.conditional)

    def class_to_image(self):
        """
        Class to image generation
        :return: Generated images
        1. If class name is `-1`, the model would output one image per class.
        2. If class name is not `-1`, the model would output the specified number of images for the specified class.
        3. The setting range should be [0, num_classes - 1].
        """
        if self.class_name == -1:
            y = torch.arange(self.num_classes).long().to(self.device)
            self.num_images = self.num_classes
        else:
            y = torch.Tensor([self.class_name] * self.num_images).long().to(self.device)
        x = self.diffusion.sample(model=self.model, n=self.num_images, x=self.input_image, labels=y,
                                  cfg_scale=self.cfg_scale)
        return x

    def text_to_image(self):
        """
        Text to image generation (coming soon)
        :return: Generated images
        """
        return None

    def generate(self, index=0, image=None, text=None):
        """
        Generate images
        :param index: Image index
        :param image: Input image tensor
        :param text: Input text description
        :return: Generated images
        """
        self.input_image, self.input_text = image, text
        if self.conditional:
            if self.mode == "class":
                x = self.class_to_image()
            elif self.mode == "text":
                x = self.text_to_image()
            else:
                raise ValueError("The mode is not supported. Please set the mode to 'class' or 'text'.")
        else:
            x = self.diffusion.sample(model=self.model, n=self.num_images, x=self.input_image)

        # If deploy app is true, return the generate results
        if self.deploy:
            return x

        if self.result_path == "" or self.result_path is None:
            plot_images(images=x)
        else:
            save_name = f"{self.generate_name}_{index}"
            save_images(images=x, path=os.path.join(self.result_path, f"{save_name}.{self.image_format}"))
            save_one_image_in_images(images=x, path=self.result_path, generate_name=save_name,
                                     image_size=self.new_image_size, image_format=self.image_format)
            # plot_images(images=x)
        logger.info(msg="Finish generation.")


def init_generate_args():
    # Generating model parameters
    # required: Must be set
    # needed: Set as needed
    # recommend: Recommend to set
    parser = argparse.ArgumentParser()
    # =================================Base settings=================================
    # Generation name (required)
    parser.add_argument("--generate_name", "-n", type=str, default="df")
    # Enable latent diffusion model (needed)
    # If enabled, the model will use latent diffusion.
    parser.add_argument("--latent", "-l", default=False, action="store_true")
    # Generated image format
    # Recommend to use png for better generation quality.
    # Option: jpg/png
    parser.add_argument("--image_format", type=str, default="png", choices=image_format_choices)
    # Number of generation images (required)
    # if class name is `-1` and conditional `is` True, the model would output one image per class.
    parser.add_argument("--num_images", type=int, default=8)
    # Use ema model
    # If set to false, the pt file of the ordinary model will be used
    # If true, the pt file of the ema model will be used
    parser.add_argument("--use_ema", default=False, action="store_true")
    # Weight path (required)
    parser.add_argument("--weight_path", type=str, default="/your/path/Defect-Diffusion-Model/weight/model.pt")
    # Set the autoencoder checkpoint path (needed)
    parser.add_argument("--autoencoder_ckpt", type=str, default="/your/path/Defect-Diffusion-Model/weight/autoencoder.pt")
    # Saving path (required)
    parser.add_argument("--result_path", type=str, default="/your/path/Defect-Diffusion-Model/results/vis")
    # Set the sample type (required)
    # If not set, the default is for 'ddpm'. You can set it to either 'ddpm', 'ddim' or 'plms'.
    # Option: ddpm/ddim/plms
    parser.add_argument("--sample", type=str, default="ddpm", choices=sample_choices)
    # Input image size (required)
    # [Warn] Compatible with older versions
    # [Warn] Version <= 1.1.1 need to be equal to model's image size, version > 1.1.1 can set whatever you want
    parser.add_argument("--image_size", "-i", type=parse_image_size_type, default=64)
    # Set the use GPU in generate (required)
    parser.add_argument("--use_gpu", type=int, default=0)

    # =====================Enable the conditional generation (if '--conditional' is set to 'True')=====================
    # Select the generation mode
    # class: Category-guided generation
    # text: Text-guided generation
    # Option: class/text
    parser.add_argument("--mode", "-m", type=str, default="class", choices=generate_mode_choices)
    # ===========** Class to Image **===========
    # Class name (required)
    # if class name is `-1`, the model would output one image per class.
    # [Note] The setting range should be [0, num_classes - 1].
    parser.add_argument("--class_name", type=int, default=-1)
    # ===========** Text to Image (coming soon) **===========
    # Text description for generation
    parser.add_argument("--text", type=str, default="")
    # ===========** Conditional generation common setting **===========
    # classifier-free guidance interpolation weight, users can better generate model effect (recommend)
    parser.add_argument("--cfg_scale", type=int, default=3)

    # =====================Older versions(version <= 1.1.1)=====================
    # Enable conditional generation (required)
    # If enabled, you can modify the custom configuration.
    # For more details, please refer to the boundary line at the bottom.
    # [Note] The conditional settings are consistent with the loaded model training settings.
    # [Warn] Compatible with older versions, version <= 1.1.1
    # [Warn] Version <= 1.1.1 need to be equal to model's network, version > 1.1.1 can set whatever you want
    parser.add_argument("--conditional", "-c", default=False, action="store_true")
    # Set network
    # Option: unet/cspdarkunet
    # [Warn] Compatible with older versions, version <= 1.1.1
    # [Warn] Version <= 1.1.1 need to be equal to model's network, version > 1.1.1 can set whatever you want
    parser.add_argument("--network", type=str, default="unet", choices=network_choices)
    # Set activation function (needed)
    # [Note] The activation function settings are consistent with the loaded model training settings.
    # [Note] If you do not set the same activation function as the model, mosaic phenomenon will occur.
    # Option: gelu/silu/relu/relu6/lrelu
    # [Warn] Compatible with older versions, version <= 1.1.1
    # [Warn] Version <= 1.1.1 need to be equal to model's act, version > 1.1.1 can set whatever you want
    parser.add_argument("--act", type=str, default="gelu", choices=act_choices)
    # Number of classes (required)
    # [Note] The classes settings are consistent with the loaded model training settings.
    # [Warn] Compatible with older versions, version <= 1.1.1
    # [Warn] Version <= 1.1.1 need to be equal to model's num classes, version > 1.1.1 can set whatever you want
    parser.add_argument("--num_classes", type=int, default=10)

    return parser.parse_args()


if __name__ == "__main__":
    # Init generate args
    args = init_generate_args()
    # Get version banner
    get_version_banner()
    gen_model = Generator(gen_args=args, deploy=False)
    for i in range(2):
        gen_model.generate(index=i)
