#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Date   : 2024/11/3 17:52
    @Author : chairc
    @Site   : https://github.com/chairc
"""
import math

from skimage.metrics import structural_similarity


def compute_psnr(mse):
    """
    PSNR
    """
    # Results
    if mse == 0:
        return 100
    else:
        return 20 * math.log10(255.0 / math.sqrt(mse))


def compute_ssim(image_outputs, image_sources):
    """
    SSIM
    """
    # Transfer to numpy
    image_outputs = image_outputs.to("cpu").numpy()
    image_sources = image_sources.to("cpu").numpy()
    ssim_list = []
    if image_outputs.shape != image_sources.shape or image_outputs.shape[0] != image_sources.shape[0]:
        raise AssertionError("Image outputs and image sources shape mismatch.")
    # image_outputs.shape[0] and image_sources.shape[0] are equal
    length = image_outputs.shape[0]
    for i in range(length):
        ssim = structural_similarity(image_outputs[i], image_sources[i], channel_axis=0, data_range=255)
        ssim_list.append(ssim)
    return sum(ssim_list) / length
