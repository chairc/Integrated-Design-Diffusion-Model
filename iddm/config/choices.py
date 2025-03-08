#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Date   : 2024/2/19 20:32
    @Author : chairc
    @Site   : https://github.com/chairc
"""
import ast
import logging

import coloredlogs

logger = logging.getLogger(__name__)
coloredlogs.install(level="INFO")

# Choice settings
# Support option
bool_choices = [True, False]
sample_choices = ["ddpm", "ddim", "plms"]
network_choices = ["unet", "cspdarkunet", "unetv2"]
optim_choices = ["adam", "adamw", "sgd"]
act_choices = ["gelu", "silu", "relu", "relu6", "lrelu"]
lr_func_choices = ["linear", "cosine", "warmup_cosine"]
image_format_choices = ["png", "jpg", "jpeg", "webp", "tif"]
noise_schedule_choices = ["linear", "cosine", "sqrt_linear", "sqrt"]
loss_func_choices = ["mse", "l1", "huber", "smooth_l1"]
sr_loss_func_choices = ["mse"]
sr_network_choices = ["srv1"]

image_type_choices = {"RGB": 3, "GRAY": 1}


# Function
def parse_image_size_type(image_size_str):
    """
    Parse image size string and return image size type
    :param image_size_str: Image size string
    :return: Image size type
    """
    # Try converting input string to integer
    logger.info(msg=f"[Note]: Input image size string is {image_size_str}.")
    try:
        image_size_int = int(image_size_str)
        if isinstance(image_size_int, int):
            image_size_int_list = [image_size_int, image_size_int]
            logger.info(msg=f"[Note]: Integer {image_size_str} converted to list {image_size_int_list}.")
            return image_size_int_list
    except ValueError:
        # If conversion to integer is not possible, try parsing to list or tuple
        parts = image_size_str.strip("[]()").split(",")
        # Check the split item is digit and length is 2
        if all(item.isdigit() for item in parts) and len(parts) == 2:
            parsed = ast.literal_eval(node_or_string=image_size_str)
            if isinstance(parsed, list) or isinstance(parsed, tuple):
                # Try converting to a list of integers
                image_size_list_and_tuple = list(map(int, parts))
                logger.info(msg=f"[Note]: {image_size_str} converted to list {image_size_list_and_tuple}.")
                return image_size_list_and_tuple
            else:
                pass
        else:
            # Throws an error if part is not a number
            raise TypeError(f"Invalid '--image_size' format: {image_size_str}")
