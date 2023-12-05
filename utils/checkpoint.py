#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Date   : 2023/12/2 22:43
    @Author : chairc
    @Site   : https://github.com/chairc
"""
import os
import numpy as np
import logging
import torch
import shutil
import coloredlogs

from collections import OrderedDict

logger = logging.getLogger(__name__)
coloredlogs.install(level="INFO")


def load_ckpt(ckpt_path, model, device, optimizer=None, is_train=True):
    """
    Load checkpoint weight files
    :param ckpt_path: Checkpoint path
    :param model: Network
    :param optimizer: Optimizer
    :param device: GPU or CPU
    :param is_train: Whether to train mode
    :return: start_epoch + 1
    """
    # Load checkpoint
    ckpt_state = torch.load(f=ckpt_path, map_location=device)
    logger.info(msg=f"[{device}]: Successfully load checkpoint, path is '{ckpt_path}'.")
    # Load the current model
    ckpt_model = ckpt_state["model"]
    load_model_ckpt(model=model, model_ckpt=ckpt_model, is_train=is_train)
    logger.info(msg=f"[{device}]: Successfully load model checkpoint.")
    # Train mode
    if is_train:
        # Load the previous model optimizer
        optim_weights_dict = ckpt_state["optimizer"]
        optimizer.load_state_dict(state_dict=optim_weights_dict)
        logger.info(msg=f"[{device}]: Successfully load optimizer checkpoint.")
        # Current checkpoint epoch
        start_epoch = ckpt_state["start_epoch"]
        # Next epoch
        return start_epoch + 1


def load_model_ckpt(model, model_ckpt, is_train=True):
    """
    Initialize weight loading
    :param model: Model
    :param model_ckpt: Model checkpoint
    :param is_train: Whether to train mode
    :return: None
    """
    model_dict = model.state_dict()
    model_weights_dict = model_ckpt
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


def save_ckpt(epoch, save_name, ckpt_model, ckpt_ema_model, ckpt_optimizer, results_dir, save_model_interval,
              start_model_interval, num_classes, classes_name=None, **kwargs):
    """
    Save the model checkpoint weight files
    :param epoch: Current epoch
    :param save_name: Save the model's name
    :param ckpt_model: Model
    :param ckpt_ema_model: EMA model
    :param ckpt_optimizer: Optimizer
    :param results_dir: Results dir
    :param save_model_interval: Whether to save weight each training
    :param start_model_interval: Start epoch for saving models
    :param num_classes: Number of classes
    :param classes_name: All classes name
    :return: None
    """
    # Checkpoint
    ckpt_state = {
        "start_epoch": epoch,
        "model": ckpt_model,
        "ema_model": ckpt_ema_model,
        "optimizer": ckpt_optimizer,
        "num_classes": num_classes,
        "classes_name": classes_name,
    }
    # Save last checkpoint, it must be done
    last_filename = os.path.join(results_dir, f"ckpt_last.pt")
    torch.save(obj=ckpt_state, f=last_filename)
    logger.info(msg=f"Save the ckpt_last.pt")
    # If save each checkpoint, just copy the last saved checkpoint and rename it
    if save_model_interval and epoch > start_model_interval:
        filename = os.path.join(results_dir, f"{save_name}.pt")
        shutil.copyfile(last_filename, filename)
        logger.info(msg=f"Save the {save_name}.pt")
    logger.info(msg="Finish saving the model.")
