#!/usr/bin/env python
# -*- coding:utf-8 -*-

# Copyright 2025 IDDM Authors

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
    @Date   : 2024/4/14 17:31
    @Author : chairc
    @Site   : https://github.com/chairc
"""
import os
import logging
import coloredlogs

import torch
import torch.nn as nn

from urllib.parse import urlparse

from iddm.config.setting import DEFAULT_IMAGE_SIZE

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


def check_is_nan(tensor):
    """
    Check the tensor is nan
    :param tensor: Tensor
    :return: True or False
    """
    if torch.isnan(input=tensor).any():
        logger.error(msg="The output tensor is nan.")
        return True
    else:
        return False


def check_path_is_exist(path):
    """
    Check the path is existed
    :param path: Path
    :return: None
    """
    # Pytorch version > 2.0.0 error: TypeError: nt._path_exists() takes no keyword arguments
    if not os.path.exists(path):
        raise FileNotFoundError(f"The path '{path}' does not exist.")


def check_image_size(image_size):
    """
    Check image size is correct and convert it to list
    :param image_size: Image size
    :return: list
    """
    if image_size is None:
        return DEFAULT_IMAGE_SIZE
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


def check_url(url=""):
    """
    Check the url is valid
    :param url: Url
    """
    try:
        # Parse URL
        parsed_url = urlparse(url)
        # Check that all parts of the parsed URL make sense
        # Here we mainly check whether the network location part (netloc) is empty
        # And whether the URL scheme is a common network protocol (such as http, https, etc.)
        if all([parsed_url.scheme, parsed_url.netloc]):
            file_name = parsed_url.path.split("/")[-1]
            logger.info(msg=f"The URL: {url} is legal.")
            return file_name
        else:
            raise ValueError(f"Invalid 'url' format: {url}")
    except ValueError:
        # If a Value Error exception is thrown when parsing the URL, it means that the URL format is illegal.
        raise ValueError("Invalid 'url' format.")


def check_pretrain_path(pretrain_path):
    """
    Check the pretrain path is valid
    :param pretrain_path: Pretrain path
    :return: Boolean
    """
    if pretrain_path is None or not os.path.exists(pretrain_path):
        return True
    return False


def check_is_distributed(distributed):
    """
    Check the distributed is valid
    :param distributed: Distributed
    :return: Boolean
    """
    if distributed and torch.cuda.device_count() > 1 and torch.cuda.is_available():
        return True
    return False


def check_gpu_num_is_valid(gpu_list):
    """
    Check the GPU number is valid
    :param gpu_list: GPU list
    :return: Boolean
    """
    len_gpu = len(gpu_list)
    # Get the total gpu number
    gpus = torch.cuda.device_count()
    # Check the gpu number is valid
    if len_gpu != 0 and len_gpu <= gpus:
        return True
    return False


def check_model_support_compile(model: nn.Module, **compile_kwargs) -> nn.Module:
    """
    Check if the model supports torch.compile and apply it if supported.
    :param model: The PyTorch model to check and potentially compile.
    :param compile_kwargs: Additional keyword arguments to pass to torch.compile.
    :return: The compiled model if supported, otherwise the original model.
    """
    # Parse the version number and compare
    # Remove possible suffixes (e.g. +cu118)
    torch_version = torch.__version__.split("+")[0]
    major, minor = map(int, torch_version.split("."))[:2]

    if major > 2 or (major == 2 and minor >= 2):
        logger.info(
            msg=f"PyTorch version {torch.__version__} supports torch.compile, enables compilation optimization...")
        # Compile the model with default parameters, and you can pass in custom parameters through kwargs
        default_kwargs = {"mode": "reduce-overhead", "dynamic": False}
        default_kwargs.update(compile_kwargs)
        return torch.compile(model, **default_kwargs)
    else:
        logger.warning(msg=f"PyTorch version {torch.__version__} < 2.2, torch.compile is not enabled.")
        return model


def check_package_is_exist(package_name):
    """
    Check the package is installed
    :param package_name: Package name
    :return: Boolean
    """
    try:
        __import__(package_name)
        logger.info(msg=f"The package '{package_name}' is installed.")
        return True
    except ImportError:
        logger.error(msg=f"The package '{package_name}' is not installed.")
        return False
