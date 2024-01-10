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


def network_initializer(network, device):
    """
    Initialize base network
    :param network: Network name
    :param device: GPU or CPU
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


def amp_initializer(amp, device):
    """
    Initialize automatic mixed precision
    :param amp: Enable automatic mixed precision
    :param device: GPU or CPU
    :return: scaler
    """
    if amp:
        logger.info(msg=f"[{device}]: Fp16 and fp32 mixed training is opened.")
        # Used to scale gradients to prevent overflow
        scaler = GradScaler()
    else:
        logger.info(msg=f"[{device}]: Fp32 training.")
        scaler = None
    return scaler


def generate_initializer(ckpt_path, args, device):
    """
    Check the parameters in checkpoint before generate
    :param ckpt_path: Checkpoint path
    :param args: Generating model parameters
    :param device: GPU or CPU
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
        if dict_params.get(param) is not None:
            logger.info(msg=f"[{device}]: Parameter {param} is exist.")
            if dict_params[param] is not None:
                logger.info(msg=f"[{device}]: Parameter {param} is not None.")
                return_param = dict_params[param]
            else:
                logger.info(msg=f"[{device}]: Parameter {param} is not None.")
                return_param = args_param
        else:
            logger.info(msg=f"[{device}]: Parameter {param} is not exist.")
            return_param = args_param
        return return_param

    logger.info(msg=f"[{device}]: Checking parameters validity.")
    # Load checkpoint before generate
    ckpt_state = torch.load(f=ckpt_path, map_location=device)
    # Valid
    conditional = check_param_in_dict(param="conditional", dict_params=ckpt_state, args_param=args.conditional)
    sample = check_param_in_dict(param="sample", dict_params=ckpt_state, args_param=args.sample)
    network = check_param_in_dict(param="network", dict_params=ckpt_state, args_param=args.network)
    image_size = check_param_in_dict(param="image_size", dict_params=ckpt_state, args_param=args.image_size)
    num_classes = check_param_in_dict(param="num_classes", dict_params=ckpt_state, args_param=args.num_classes)
    act = check_param_in_dict(param="act", dict_params=ckpt_state, args_param=args.act)
    logger.info(msg=f"[{device}]: Successfully checked parameters.")
    return conditional, sample, network, image_size, num_classes, act
