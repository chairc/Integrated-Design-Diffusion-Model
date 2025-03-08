#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Date   : 2024/1/27 15:33
    @Author : chairc
    @Site   : https://github.com/chairc
"""
import torch
import torchvision
import logging
import coloredlogs

from iddm.config.setting import SR_MEAN, SR_STD
from iddm.utils.checkpoint import load_ckpt
from iddm.utils.initializer import sr_network_initializer, device_initializer
from iddm.utils.utils import download_model_pretrain_model

logger = logging.getLogger(__name__)
coloredlogs.install(level="INFO")


def prepare_image(images):
    """
    Prepare images
    :param images: Images
    :return: images
    """
    transforms = torchvision.transforms.Compose([
        # To Tensor Format
        torchvision.transforms.ToTensor(),
        # For standardization, the mean and standard deviation
        torchvision.transforms.Normalize(mean=SR_MEAN, std=SR_STD)
    ])
    images = transforms(images).unsqueeze(dim=0)
    return images


def post_image(images, device="cpu"):
    """
    Post images
    :param images: Images
    :param device: CPU or GPU
    :return: new_images
    """
    mean_tensor = torch.tensor(data=SR_MEAN).view(1, -1, 1, 1).to(device)
    std_tensor = torch.tensor(data=SR_STD).view(1, -1, 1, 1).to(device)
    new_images = images * std_tensor + mean_tensor
    # Limit the image between 0 and 1
    new_images = (new_images.clamp(0, 1) * 255).to(torch.uint8)
    return new_images


def load_sr_model(enable_custom=False, weight_path=None):
    """
    Load super resolution model
    :param enable_custom: Whether to enable custom model
    :param weight_path: Super resolution model weight path
    :return: model
    """
    # Custom weight model path
    if enable_custom:
        default_weight_path = weight_path
    # If weight model not fill, download it in the official GitHub
    else:
        default_weight_path = download_model_pretrain_model(pretrain_type="sr")

    # Get the device information
    device = device_initializer()
    # Check the weight model
    ckpt_state = torch.load(f=default_weight_path, map_location=device)
    network = ckpt_state["network"]
    image_size = ckpt_state["image_size"]
    act = ckpt_state["act"]
    logger.info(msg=f"Current load model parameters [network: {network}, image_size: {image_size}, "
                    f"act: {act}]")
    # Check parameters is not None
    assert network is not None and image_size is not None and act is not None, \
        "Weight model's parameters should not None."
    # Enable network, image size and select activation function
    Network = sr_network_initializer(network=network, device=device)
    # Init model
    model = Network(act=act).to(device)
    # Load model
    load_ckpt(ckpt_path=default_weight_path, model=model, device=device, is_train=False)
    model.eval()
    return model


def inference(image, model, device):
    """
    Low resolution image to high resolution
    :param image: Image
    :param model: Model
    :param device: GPU or CPU
    :return: result
    """
    image = prepare_image(image)
    image = image.to(device)
    # Infer
    with torch.no_grad():
        output = model(image)
    result = post_image(output)
    return result
