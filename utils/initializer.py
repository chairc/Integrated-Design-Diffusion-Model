#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Date   : 2023/6/20 19:05
    @Author : chairc
    @Site   : https://github.com/chairc
"""
import random
import numpy as np
import torch
import logging
import coloredlogs

from collections import OrderedDict

from torch.cuda.amp import GradScaler

from model.networks.unet import UNet
from model.networks.cspdarkunet import CSPDarkUnet
from model.samples.ddim import DDIMDiffusion
from model.samples.ddpm import DDPMDiffusion
from utils.lr_scheduler import set_cosine_lr

logger = logging.getLogger(__name__)
coloredlogs.install(level="INFO")


def device_initializer(device_id=0):
    """
    This function initializes the running device information when the program runs for the first time
    :return: cpu or cuda
    """
    logger.info(msg="Init program, it is checking the basic device setting.")
    device_dict = {}
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        # Set device with custom setting
        device = torch.device("cuda", device_id)
        is_init = torch.cuda.is_initialized()
        device_count = torch.cuda.device_count()
        device_name = torch.cuda.get_device_name(device=device)
        device_cap = torch.cuda.get_device_capability(device=device)
        device_prop = torch.cuda.get_device_properties(device=device)
        device_dict["device_id"] = device_id
        device_dict["is_init"] = is_init
        device_dict["device_count"] = device_count
        device_dict["device_name"] = device_name
        device_dict["device_cap"] = device_cap
        device_dict["device_prop"] = device_prop
        logger.info(msg=device_dict)
    else:
        logger.warning(msg="Warning: The device is using cpu, the device would slow down the model running speed.")
        device = torch.device(device="cpu")
    return device


def seed_initializer(seed_id=0):
    """
    Initialize the seed
    :param seed_id: The seed id
    :return: None
    """
    torch.manual_seed(seed_id)
    torch.cuda.manual_seed_all(seed_id)
    random.seed(seed_id)
    np.random.seed(seed_id)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(msg=f"The seed is initialized, and the seed ID is {seed_id}.")


def load_model_weight_initializer(model, weight_path, device, is_train=True):
    """
    Initialize weight loading
    :param model: Model
    :param weight_path: Weight model path
    :param device: GPU or CPU
    :param is_train: Whether to train mode
    :return: None
    """
    model_dict = model.state_dict()
    model_weights_dict = torch.load(f=weight_path, map_location=device)
    # Check if key contains 'module.' prefix.
    # This method is the name after training in the distribution, check the weight and delete
    if not is_train:
        new_model_weights_dict = {}
        for key, value in model_weights_dict.items():
            if key.startswith("module."):
                new_key = key[len("module."):]
                new_model_weights_dict[new_key] = value
            else:
                new_model_weights_dict[key] = value
        model_weights_dict = new_model_weights_dict
        logger.info(msg="Successfully check the load weight and rename.")
    model_weights_dict = {k: v for k, v in model_weights_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
    model_dict.update(model_weights_dict)
    model.load_state_dict(state_dict=OrderedDict(model_dict))


def network_initializer(network, device):
    """
    Initialize base network
    :param network: Network name
    :return: Network
    """
    if network == "unet":
        Network = UNet
    elif network == "cspdarkunet":
        Network = CSPDarkUnet
    else:
        Network = UNet
        logger.warning(msg=f"[{device}]: Setting network error, we has been automatically set to unet.")
    logger.info(msg=f"[{device}]: This base network is {network}")
    return Network


def optimizer_initializer(model, optim, init_lr, device):
    """
    Initialize optimizer
    :param model: Model
    :param optim: Optimizer name
    :param init_lr: Initialize learning rate
    :return: optimizer
    """
    if optim == "adam":
        optimizer = torch.optim.Adam(params=model.parameters(), lr=init_lr)
    elif optim == "adamw":
        optimizer = torch.optim.AdamW(params=model.parameters(), lr=init_lr)
    else:
        optimizer = torch.optim.AdamW(params=model.parameters(), lr=init_lr)
        logger.warning(msg=f"[{device}]: Setting optimizer error, we has been automatically set to adamw.")
    logger.info(msg=f"[{device}]: This base optimizer is {optim}")
    return optimizer


def sample_initializer(sample, image_size, device):
    """
    Initialize sample
    :param sample: Sample function
    :param image_size: image size
    :param device: GPU or CPU
    :return: diffusion
    """
    if sample == "ddpm":
        diffusion = DDPMDiffusion(img_size=image_size, device=device)
    elif sample == "ddim":
        diffusion = DDIMDiffusion(img_size=image_size, device=device)
    else:
        diffusion = DDPMDiffusion(img_size=image_size, device=device)
        logger.warning(msg=f"[{device}]: Setting sample error, we has been automatically set to ddpm.")
    return diffusion


def lr_initializer(lr_func, optimizer, epoch, epochs, init_lr, device):
    """
    Initialize learning rate
    :param lr_func: learning rate function
    :param optimizer: Optimizer
    :param epoch: Current epoch
    :param epochs: Total epoch
    :param init_lr: Initialize learning rate
    :param device: GPU or CPU
    :return: current_lr
    """
    if lr_func == "cosine":
        current_lr = set_cosine_lr(optimizer=optimizer, current_epoch=epoch, max_epoch=epochs,
                                   lr_min=init_lr * 0.01, lr_max=init_lr, warmup=False)
    elif lr_func == "warmup_cosine":
        current_lr = set_cosine_lr(optimizer=optimizer, current_epoch=epoch, max_epoch=epochs,
                                   lr_min=init_lr * 0.01, lr_max=init_lr, warmup=True)
    else:
        current_lr = init_lr
    logger.info(msg=f"[{device}]: This epoch learning rate is {current_lr}")
    return current_lr


def fp16_initializer(fp16, device):
    """
    Initialize harf-precision
    :param fp16: harf-precision
    :param device: GPU or CPU
    :return: scaler
    """
    if fp16:
        logger.info(msg=f"[{device}]: Fp16 training is opened.")
        # Used to scale gradients to prevent overflow
        scaler = GradScaler()
    else:
        logger.info(msg=f"[{device}]: Fp32 training.")
        scaler = None
    return scaler
