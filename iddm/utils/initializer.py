#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Date   : 2023/6/20 19:05
    @Author : chairc
    @Site   : https://github.com/chairc
"""
import os
import random
import numpy as np
import torch
import torch.nn as nn
import logging
import coloredlogs

from torch.cuda.amp import GradScaler

from iddm.model.networks.unet import UNet
from iddm.model.networks.unetv2 import UNetV2
from iddm.model.networks.cspdarkunet import CSPDarkUnet
from iddm.model.networks.sr.srv1 import SRv1
from iddm.model.samples.ddim import DDIMDiffusion
from iddm.model.samples.ddpm import DDPMDiffusion
from iddm.model.samples.plms import PLMSDiffusion
from iddm.utils.check import check_path_is_exist
from iddm.utils.lr_scheduler import set_cosine_lr

logger = logging.getLogger(__name__)
coloredlogs.install(level="INFO")


def device_initializer(device_id=0, is_train=False):
    """
    This function initializes the running device information when the program runs for the first time
    [Warn] This project will no longer support CPU training after v1.1.2
    :param device_id: Device id
    :param is_train: Whether to train mode
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
        return device
    else:
        logger.warning(msg="This project will no longer support CPU training after version 1.1.2")
        if is_train:
            raise NotImplementedError("CPU training is no longer supported after version 1.1.2")
        else:
            # Generate or test mode
            logger.warning(msg="Warning: The device is using cpu, the device would slow down the model running speed.")
            return torch.device(device="cpu")


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


def network_initializer(network, device):
    """
    Initialize base network
    :param network: Network name
    :param device: GPU or CPU
    :return: Network
    """
    if network == "unet":
        Network = UNet
    elif network == "unetv2":
        Network = UNetV2
    elif network == "cspdarkunet":
        Network = CSPDarkUnet
    else:
        Network = UNet
        logger.warning(msg=f"[{device}]: Setting network error, we has been automatically set to unet.")
    logger.info(msg=f"[{device}]: This base network is {network}")
    return Network


def sr_network_initializer(network, device):
    """
    Initialize super resolution network
    :param network: Network name
    :param device: GPU or CPU
    :return: Network
    """
    if network == "srv1":
        Network = SRv1
    else:
        Network = SRv1
        logger.warning(msg=f"[{device}]: Setting network error, we has been automatically set to srv1.")
    logger.info(msg=f"[{device}]: This super resolution network is {network}")
    return Network


def loss_initializer(loss_name, device):
    """
    Initialize loss function
    :param loss_name: Loss function name
    :param device: GPU or CPU
    :return: Network
    """
    if loss_name == "mse":
        loss_function = nn.MSELoss()
    elif loss_name == "l1":
        loss_function = nn.L1Loss()
    elif loss_name == "huber":
        loss_function = nn.HuberLoss()
    elif loss_name == "smooth_l1":
        loss_function = nn.SmoothL1Loss()
    else:
        loss_function = nn.MSELoss()
        logger.warning(msg=f"[{device}]: Setting loss function error, we has been automatically set to mse loss.")
    logger.info(msg=f"[{device}]: This loss function is {loss_name}")
    return loss_function


def optimizer_initializer(model, optim, init_lr, device):
    """
    Initialize optimizer
    :param model: Model
    :param optim: Optimizer name
    :param init_lr: Initialize learning rate
    :param device: GPU or CPU
    :return: optimizer
    """
    # Set model parameters
    model_param = model.parameters()
    # Choose an optimizer
    if optim == "adam":
        optimizer = torch.optim.Adam(params=model_param, lr=init_lr)
    elif optim == "adamw":
        optimizer = torch.optim.AdamW(params=model_param, lr=init_lr)
    elif optim == "sgd":
        optimizer = torch.optim.SGD(params=model_param, lr=init_lr, momentum=0.937)
    else:
        optimizer = torch.optim.AdamW(params=model_param, lr=init_lr)
        logger.warning(msg=f"[{device}]: Setting optimizer error, we has been automatically set to adamw.")
    logger.info(msg=f"[{device}]: This base optimizer is {optim}")
    return optimizer


def sample_initializer(sample, image_size, device, schedule_name="linear"):
    """
    Initialize sample
    :param sample: Sample function
    :param image_size: image size
    :param device: GPU or CPU
    :param schedule_name: Prepare the noise schedule name
    :return: diffusion
    """
    if sample == "ddpm":
        diffusion = DDPMDiffusion(img_size=image_size, device=device, schedule_name=schedule_name)
    elif sample == "ddim":
        diffusion = DDIMDiffusion(img_size=image_size, device=device, schedule_name=schedule_name)
    elif sample == "plms":
        diffusion = PLMSDiffusion(img_size=image_size, device=device, schedule_name=schedule_name)
    else:
        diffusion = DDPMDiffusion(img_size=image_size, device=device, schedule_name=schedule_name)
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


def amp_initializer(amp, device):
    """
    Initialize automatic mixed precision
    :param amp: Enable automatic mixed precision
    :param device: GPU or CPU
    :return: scaler
    """
    if amp:
        logger.info(msg=f"[{device}]: Automatic mixed precision training.")
    else:
        logger.info(msg=f"[{device}]: Normal training.")
    # Used to scale gradients to prevent overflow
    return GradScaler(enabled=amp)


def generate_initializer(ckpt_path, conditional, network, image_size, num_classes, act, device, **kwargs):
    """
    Check the parameters in checkpoint before generate
    :param ckpt_path: Checkpoint path
    :param conditional: Conditional
    :param network: Network
    :param image_size: Image size
    :param num_classes: Number of classes
    :param act: Activation function
    :param device: GPU or CPU
    :param kwargs: Additional arguments
    :return: [conditional, sample, network, image_size, num_classes, act]
    """

    def check_param_in_dict(param, dict_params, args_param):
        """
        Check the params in dict
        :param param: Parameter
        :param dict_params: Parameters
        :param args_param: Argparse parameter
        :return: return_param
        """
        logger.info(msg=f"[{device}]: Input parameter is {args_param}.")
        if dict_params.get(param) is not None:
            logger.info(msg=f"[{device}]: Model parameter is {dict_params[param]}.")
            logger.info(msg=f"[{device}]: Parameter {param} is exist.")
            if dict_params[param] is not None:
                logger.info(msg=f"[{device}]: Parameter {param} value is not None.")
                return_param = dict_params[param]
            else:
                logger.warning(msg=f"[{device}]: Parameter {param} value is None.")
                return_param = args_param
        else:
            logger.warning(msg=f"[{device}]: Parameter {param} is not exist.")
            return_param = args_param
        return return_param

    logger.info(msg=f"[{device}]: Checking parameters validity.")
    # Load checkpoint before generate
    ckpt_state = torch.load(f=ckpt_path, map_location=device)
    # Valid
    conditional = check_param_in_dict(param="conditional", dict_params=ckpt_state, args_param=conditional)
    network = check_param_in_dict(param="network", dict_params=ckpt_state, args_param=network)
    image_size = check_param_in_dict(param="image_size", dict_params=ckpt_state, args_param=image_size)
    num_classes = check_param_in_dict(param="num_classes", dict_params=ckpt_state, args_param=num_classes)
    act = check_param_in_dict(param="act", dict_params=ckpt_state, args_param=act)
    logger.info(msg=f"[{device}]: Successfully checked parameters.")
    return conditional, network, image_size, num_classes, act


def classes_initializer(dataset_path):
    """
    Initialize number of classes
    :param dataset_path: Dataset path
    :return: num_classes
    """
    check_path_is_exist(path=dataset_path)
    num_classes = 0
    # Loop dataset path
    for classes_dir in os.listdir(path=dataset_path):
        # Check current dir
        if os.path.isdir(s=os.path.join(dataset_path, classes_dir)):
            num_classes += 1
    logger.info(msg=f"Current number of classes is {num_classes}.")
    if num_classes == 0:
        raise Exception(f"No dataset folders found in '{dataset_path}'.")
    return num_classes
