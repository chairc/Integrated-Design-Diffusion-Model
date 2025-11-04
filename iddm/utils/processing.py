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
    @Date   : 2024/11/8 20:15
    @Author : chairc
    @Site   : https://github.com/chairc
"""
import io
import base64
import logging
import coloredlogs
import numpy as np

from torchvision import transforms

from PIL import Image

logger = logging.getLogger(__name__)
coloredlogs.install(level="INFO")


def base64_to_image(base64_stream=None):
    """
    Transforms base64 image to PIL image
    :param base64_stream: base64 image
    """
    # Decode Base 64 string into bytes
    image_byte = base64.b64decode(s=base64_stream)
    # Load images using byte stream
    image = Image.open(io.BytesIO(initial_bytes=image_byte))
    return image


def image_to_base64(image=None, image_name=None, encoding="utf-8"):
    """
    Transforms PIL image to base64 image
    :param image: PIL image
    :param image_name: base64 image name
    :param encoding: base64 image encoding
    """
    if image is None:
        raise TypeError("Image is required.")
    if image_name is None:
        image_format = "png"
    else:
        image_format = image_name.split(".")[-1]
    image_buffer = io.BytesIO()
    image.save(image_buffer, format=image_format)
    image_base64 = base64.b64encode(s=image_buffer.getvalue()).decode(encoding=encoding)
    return image_base64


def pil_image_to_tensor(image):
    """
    Convert a PIL image to a PyTorch tensor.
    :param image: PIL Image
    :return: PyTorch tensor
    """
    if not isinstance(image, Image.Image):
        logger.error("Image must be PIL Image")
        raise TypeError("Input must be a PIL image.")
    # Convert the PIL image to a PyTorch tensor
    image_tensor = transforms.ToTensor()(image)
    return image_tensor


def numpy_image_to_tensor(image):
    """
    Convert a NumPy image to a PyTorch tensor.
    :param image: NumPy array
    :return: PyTorch tensor
    """
    if not isinstance(image, (np.ndarray,)):
        logger.error("Image must be NumPy array")
        raise TypeError("Input must be a NumPy array.")
    # Convert the NumPy array to a PyTorch tensor
    image_tensor = transforms.ToTensor()(Image.fromarray(image))
    return image_tensor


def tensor_to_pil_image(tensor):
    """
    Convert a PyTorch tensor to a PIL image.
    :param tensor: PyTorch tensor
    :return: PIL Image
    """
    if not isinstance(tensor, (np.ndarray,)):
        logger.error("Input must be a NumPy array")
        raise TypeError("Input must be a NumPy array.")
    # Convert the PyTorch tensor to a PIL image
    image = transforms.ToPILImage()(tensor)
    return image


def tensor_to_numpy_image(tensor):
    """
    Convert a PyTorch tensor to a NumPy image.
    :param tensor: PyTorch tensor
    :return: NumPy array
    """
    if not isinstance(tensor, (np.ndarray,)):
        logger.error("Input must be a NumPy array")
        raise TypeError("Input must be a NumPy array.")
    # Convert the PyTorch tensor to a NumPy array
    image = np.array(transforms.ToPILImage()(tensor))
    return image
