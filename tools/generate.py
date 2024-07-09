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
from config.choices import sample_choices, network_choices, act_choices, image_format_choices, parse_image_size_type
from utils.check import check_image_size
from utils.initializer import device_initializer, network_initializer, sample_initializer, generate_initializer
from utils.utils import plot_images, save_images, save_one_image_in_images, check_and_create_dir
from utils.checkpoint import load_ckpt

logger = logging.getLogger(__name__)
coloredlogs.install(level="INFO")


def generate(args):
    """
    Generating
    :param args: Input parameters
    :return: None
    """
    logger.info(msg="Start generation.")
    logger.info(msg=f"Input params: {args}")
    # Weight path
    weight_path = args.weight_path
    # Run device initializer
    device = device_initializer()
    # Enable conditional generation, sample type, network, image size, number of classes and select activation function
    conditional, network, image_size, num_classes, act = generate_initializer(ckpt_path=weight_path, args=args,
                                                                              device=device)
    # Check image size format
    image_size = check_image_size(image_size=image_size)
    # Generation name
    generate_name = args.generate_name
    # Sample
    sample = args.sample
    # Number of images
    num_images = args.num_images
    # Use ema
    use_ema = args.use_ema
    # Format of images
    image_format = args.image_format
    # Saving path
    result_path = os.path.join(args.result_path, str(time.time()))
    # Check and create result path
    check_and_create_dir(result_path)
    # Network
    Network = network_initializer(network=network, device=device)
    # Initialize the diffusion model
    diffusion = sample_initializer(sample=sample, image_size=image_size, device=device)
    # Is it necessary to expand the image?
    expand_flag = args.image_size if image_size is not args.image_size else None
    # Initialize model
    if conditional:
        # Generation class name
        class_name = args.class_name
        # classifier-free guidance interpolation weight
        cfg_scale = args.cfg_scale
        model = Network(num_classes=num_classes, device=device, image_size=image_size, act=act).to(device)
        load_ckpt(ckpt_path=weight_path, model=model, device=device, is_train=False, is_use_ema=use_ema,
                  conditional=conditional)
        if class_name == -1:
            y = torch.arange(num_classes).long().to(device)
            num_images = num_classes
        else:
            y = torch.Tensor([class_name] * num_images).long().to(device)
        x = diffusion.sample(model=model, n=num_images, labels=y, cfg_scale=cfg_scale)
    else:
        model = Network(device=device, image_size=image_size, act=act).to(device)
        load_ckpt(ckpt_path=weight_path, model=model, device=device, is_train=False, conditional=conditional)
        x = diffusion.sample(model=model, n=num_images)
    # If there is no path information, it will only be displayed
    # If it exists, it will be saved to the specified path and displayed
    if result_path == "" or result_path is None:
        plot_images(images=x)
    else:
        save_images(images=x, path=os.path.join(result_path, f"{generate_name}.{image_format}"))
        save_one_image_in_images(images=x, path=result_path, generate_name=generate_name, image_size=expand_flag,
                                 image_format=image_format)
        plot_images(images=x)
    logger.info(msg="Finish generation.")


if __name__ == "__main__":
    # Generating model parameters
    # required: Must be set
    # needed: Set as needed
    # recommend: Recommend to set
    parser = argparse.ArgumentParser()
    # =================================Base settings=================================
    # Generation name (required)
    parser.add_argument("--generate_name", type=str, default="df")
    # Input image size (required)
    # [Warn] Compatible with older versions
    # [Warn] Version <= 1.1.1 need to be equal to model's image size, version > 1.1.1 can set whatever you want
    parser.add_argument("--image_size", type=parse_image_size_type, default=64)
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
    # Saving path (required)
    parser.add_argument("--result_path", type=str, default="/your/path/Defect-Diffusion-Model/results/vis")
    # Set the sample type (required)
    # If not set, the default is for 'ddpm'. You can set it to either 'ddpm', 'ddim' or 'plms'.
    # Option: ddpm/ddim/plms
    parser.add_argument("--sample", type=str, default="ddpm", choices=sample_choices)

    # =====================Enable the conditional generation (if '--conditional' is set to 'True')=====================
    # Class name (required)
    # if class name is `-1`, the model would output one image per class.
    # [Note] The setting range should be [0, num_classes - 1].
    parser.add_argument("--class_name", type=int, default=0)
    # classifier-free guidance interpolation weight, users can better generate model effect (recommend)
    parser.add_argument("--cfg_scale", type=int, default=3)

    # =====================Older versions(version <= 1.1.1)=====================
    # Enable conditional generation (required)
    # If enabled, you can modify the custom configuration.
    # For more details, please refer to the boundary line at the bottom.
    # [Note] The conditional settings are consistent with the loaded model training settings.
    # [Warn] Compatible with older versions, version <= 1.1.1
    # [Warn] Version <= 1.1.1 need to be equal to model's network, version > 1.1.1 can set whatever you want
    parser.add_argument("--conditional", default=False, action="store_true")
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

    args = parser.parse_args()
    generate(args)
