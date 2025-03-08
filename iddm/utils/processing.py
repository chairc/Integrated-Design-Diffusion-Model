#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Date   : 2024/11/8 20:15
    @Author : chairc
    @Site   : https://github.com/chairc
"""
import io
import base64
import logging
import coloredlogs

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
