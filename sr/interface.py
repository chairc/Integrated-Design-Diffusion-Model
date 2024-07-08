#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Date   : 2024/1/27 15:33
    @Author : chairc
    @Site   : https://github.com/chairc
"""
import os

import coloredlogs
import requests
import torch
import torchvision
import logging

from config.setting import MEAN, STD
from utils.checkpoint import load_ckpt
from utils.initializer import sr_network_initializer, device_initializer
from utils.utils import check_and_create_dir

logger = logging.getLogger(__name__)
coloredlogs.install(level="INFO")

DEFAULT_WEIGHT_ROOT_PATH = "../weights/sr"
SR_WEIGHT_NAME = "srv1"

SRv1_DOWNLOAD_URL = ""
# Key: weight name
# Value: [weight name, download url]
SR_DICT = {"srv1": ["srv1", SRv1_DOWNLOAD_URL]}

# Custom super-resolution model
# /your/sr/weight/model/path/model.pt
CUSTOM_WEIGHT_PATH = ""


def download_weight(download_path, save_path):
    """
    Download model weight
    :param download_path: Download url
    :param save_path: Save the model path
    :return: None
    """
    if download_path == "" or download_path is None:
        assert False, "[Error]: Download path is not exist, please check the download url."
    res = requests.get(url=download_path, stream=True)
    res.raise_for_status()
    # Save the weight model
    with open(file=save_path, mode="wb") as f:
        for chunk in res.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
    f.close()
    logger.info(msg=f"[Note]: Successfully downloaded super-resolution model ({SR_WEIGHT_NAME}) in official GitHub.")


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
        torchvision.transforms.Normalize(mean=MEAN, std=STD)
    ])
    images = transforms(images).unsqueeze(dim=0)
    return images


def post_image(images):
    """
    Post images
    :param images: Images
    :return: new_images
    """
    new_images = torch.empty(size=images.shape, dtype=torch.uint8)
    for i in range(images.shape[0]):
        new_image = (images[i].clamp(-1, 1) + 1) / 2
        new_image = (new_image * 255).to(torch.uint8)
        new_images[i] = new_image
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
        # Check dir
        check_and_create_dir(DEFAULT_WEIGHT_ROOT_PATH)
        # Get weight information
        weight_params = SR_DICT.get(SR_WEIGHT_NAME)
        weight_name = weight_params[0]
        weight_download_url = weight_params[1]
        default_weight_path = os.path.join(DEFAULT_WEIGHT_ROOT_PATH, f"{weight_name}.pt")

        # Checking the sr weight is exist
        if not os.path.exists(default_weight_path):
            # Download
            download_weight(download_path=weight_download_url, save_path=default_weight_path)

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
