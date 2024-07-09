#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Date   : 2024/4/14 17:31
    @Author : chairc
    @Site   : https://github.com/chairc
"""
import os
import logging
import coloredlogs

logger = logging.getLogger(__name__)
coloredlogs.install(level="INFO")


def check_and_create_dir(path):
    """
    Check and create not exist folder
    :param path: Create path
    :return: None
    """
    logger.info(msg=f"Check and create folder '{path}'.")
    os.makedirs(name=path, exist_ok=True)


def check_path_is_exist(path):
    """
    Check the path is existed
    :param path: Path
    :return: None
    """
    if not os.path.exists(path=path):
        raise FileNotFoundError(f"The path '{path}' does not exist.")


def check_image_size(image_size):
    """
    Check image size is correct and convert it to list
    :param image_size: Image size
    :return: list
    """
    if image_size is None:
        return [64, 64]
    elif isinstance(image_size, int):
        return [image_size, image_size]
    elif isinstance(image_size, tuple) or isinstance(image_size, list):
        # Check all items in list and tuple are int
        if all(isinstance(item, int) for item in image_size) and len(image_size) == 2:
            return list(image_size)
        else:
            raise ValueError(f"Invalid 'image_size' tuple and list format: {image_size}")
    else:
        raise TypeError(f"Invalid 'image_size' format: {image_size}")
