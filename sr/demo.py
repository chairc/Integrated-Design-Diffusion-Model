#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Date   : 2024/1/27 14:48
    @Author : chairc
    @Site   : https://github.com/chairc
"""
import os
import sys
import argparse
import time

import logging
import coloredlogs

from PIL import Image

sys.path.append(os.path.dirname(sys.path[0]))
from config.version import get_version_banner
from sr.interface import inference, load_sr_model
from utils.initializer import device_initializer
from utils.utils import plot_images, save_images, check_and_create_dir

logger = logging.getLogger(__name__)
coloredlogs.install(level="INFO")


def lr2hr(args):
    """
    Super resolution
    :param args: Input parameters
    :return: None
    """
    logger.info(msg="Start low resolution to high resolution.")
    logger.info(msg=f"Input params: {args}")
    # Image path
    image_path = args.image_path
    # Weight path
    weight_path = args.weight_path
    # Run device initializer
    device = device_initializer()
    # Generation name
    generate_name = args.generate_name
    sr_model_path = weight_path if weight_path != "" else None
    # Saving path
    result_path = os.path.join(args.result_path, str(time.time()))
    # Check and create result path
    check_and_create_dir(result_path)
    # Load model
    model = load_sr_model(enable_custom=True, weight_path=sr_model_path)
    # Open the image
    image = Image.open(fp=image_path)
    x = inference(image=image, model=model, device=device)
    # If there is no path information, it will only be displayed
    # If it exists, it will be saved to the specified path and displayed
    if result_path == "" or result_path is None:
        plot_images(images=x)
    else:
        save_images(images=x, path=os.path.join(result_path, f"{generate_name}.jpg"))
        plot_images(images=x)
    logger.info(msg="Finish super resolution.")


if __name__ == "__main__":
    # Low resolution to high resolution model parameters
    # required: Must be set
    # needed: Set as needed
    # recommend: Recommend to set
    parser = argparse.ArgumentParser()
    # =================================Base settings=================================
    # Generation name (required)
    parser.add_argument("--generate_name", type=str, default="sr_64_to_256")
    # Input image size (required)
    # By default, the current size is multiplied by 4 in Super resolution network
    parser.add_argument("--image_size", type=int, default=64)
    # Image path (required)
    parser.add_argument("--image_path", type=str, default="/your/path/Diffusion-Model/sr/test.jpg")
    # Weight path (required)
    # If it is empty, the official super weight model is downloaded.
    parser.add_argument("--weight_path", type=str, default="/your/path/Diffusion-Model/sr/ckpt.pt")
    # Saving path (required)
    parser.add_argument("--result_path", type=str, default="/your/path/Diffusion-Model/result")

    args = parser.parse_args()
    # Get version banner
    get_version_banner()
    lr2hr(args)
