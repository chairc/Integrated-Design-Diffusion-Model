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

import torch
import logging
import coloredlogs

sys.path.append(os.path.dirname(sys.path[0]))
from utils.initializer import device_initializer, load_model_weight_initializer, network_initializer, sample_initializer
from utils.utils import plot_images, save_images, check_and_create_dir

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
    # Enable conditional generation
    conditional = args.conditional
    # Sample type
    sample = args.sample
    # Network
    network = args.network
    # Generation name
    generate_name = args.generate_name
    # Image size
    image_size = args.image_size
    # Select activation function
    act = args.act
    # Number of images
    num_images = args.num_images
    # Weight path
    weight_path = args.weight_path
    # Saving path
    result_path = args.result_path
    # Run device initializer
    device = device_initializer()
    # Check and create result path
    check_and_create_dir(result_path)
    # Network
    Network = network_initializer(network=network, device=device)
    # Initialize the diffusion model
    diffusion = sample_initializer(sample=sample, image_size=image_size, device=device)
    # Initialize model
    if conditional:
        # Number of classes
        num_classes = args.num_classes
        # Generation class name
        class_name = args.class_name
        # classifier-free guidance interpolation weight
        cfg_scale = args.cfg_scale
        model = Network(num_classes=num_classes, device=device, image_size=image_size, act=act).to(device)
        load_model_weight_initializer(model=model, weight_path=weight_path, device=device, is_train=False)
        if class_name == -1:
            y = torch.arange(num_classes).long().to(device)
            num_images = num_classes
        else:
            y = torch.Tensor([class_name] * num_images).long().to(device)
        x = diffusion.sample(model=model, n=num_images, labels=y, cfg_scale=cfg_scale)
    else:
        model = Network(device=device, image_size=image_size, act=act).to(device)
        load_model_weight_initializer(model=model, weight_path=weight_path, device=device, is_train=False)
        x = diffusion.sample(model=model, n=num_images)
    # If there is no path information, it will only be displayed
    # If it exists, it will be saved to the specified path and displayed
    if result_path == "" or result_path is None:
        plot_images(images=x)
    else:
        save_images(images=x, path=os.path.join(result_path, f"{generate_name}.jpg"))
        plot_images(images=x)
    logger.info(msg="Finish generation.")


if __name__ == "__main__":
    # Generating model parameters
    # required: Must be set
    # needed: Set as needed
    # recommend: Recommend to set
    parser = argparse.ArgumentParser()
    # =================================Base settings=================================
    # Enable conditional generation (required)
    # If enabled, you can modify the custom configuration.
    # For more details, please refer to the boundary line at the bottom.
    # [Note] The conditional settings are consistent with the loaded model training settings.
    parser.add_argument("--conditional", type=bool, default=True)
    # Generation name (required)
    parser.add_argument("--generate_name", type=str, default="df")
    # Input image size (required)
    parser.add_argument("--image_size", type=int, default=64)
    # Number of generation images (required)
    # if class name is `-1` and conditional `is` True, the model would output one image per class.
    parser.add_argument("--num_images", type=int, default=8)
    # Weight path (required)
    parser.add_argument("--weight_path", type=str, default="/your/path/Defect-Diffusion-Model/weight/model.pt")
    # Saving path (required)
    parser.add_argument("--result_path", type=str, default="/your/path/Defect-Diffusion-Model/results/vis")
    # Set the sample type (required)
    # If not set, the default is for 'ddpm'. You can set it to either 'ddpm' or 'ddim'.
    # Option: ddpm/ddim
    parser.add_argument("--sample", type=str, default="ddpm")
    # Set network
    # Option: unet/cspdarkunet
    parser.add_argument("--network", type=str, default="unet")
    # Set activation function (needed)
    # [Note] The activation function settings are consistent with the loaded model training settings.
    # [Note] If you do not set the same activation function as the model, mosaic phenomenon will occur.
    # Option: gelu/silu/relu/relu6/lrelu
    parser.add_argument("--act", type=str, default="gelu")

    # =====================Enable the conditional generation (if '--conditional' is set to 'True')=====================
    # Number of classes (required)
    # [Note] The classes settings are consistent with the loaded model training settings.
    parser.add_argument("--num_classes", type=int, default=10)
    # Class name (required)
    # if class name is `-1`, the model would output one image per class.
    # [Note] The setting range should be [0, num_classes - 1].
    parser.add_argument("--class_name", type=int, default=0)
    # classifier-free guidance interpolation weight, users can better generate model effect (recommend)
    parser.add_argument("--cfg_scale", type=int, default=3)

    args = parser.parse_args()
    generate(args)
