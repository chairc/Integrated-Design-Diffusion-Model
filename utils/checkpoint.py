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


def load_ckpt(ckpt_path, model, device, optimizer=None, is_train=True, is_pretrain=False, is_distributed=False,
              is_use_ema=False, conditional=False):
    """
    Load checkpoint weight files
    :param ckpt_path: Checkpoint path
    :param model: Network
    :param optimizer: Optimizer
    :param device: GPU or CPU
    :param is_train: Whether to train mode
    :param is_pretrain: Whether to load pretrain checkpoint
    :param is_distributed:  Whether to distribute training
    :param is_use_ema:  Whether to use ema model or model
    :param conditional:  Whether conditional training
    :return: start_epoch + 1
    """
    # Load checkpoint
    ckpt_state = torch.load(f=ckpt_path, map_location=device)
    if is_pretrain:
        logger.info(msg=f"[{device}]: Successfully load pretrain checkpoint, path is '{ckpt_path}'.")
    else:
        logger.info(msg=f"[{device}]: Successfully load checkpoint, path is '{ckpt_path}'.")
    # Check checkpoint's structure
    assert ckpt_state["model"] is not None or ckpt_state["ema_model"] is not None, \
        "Error!! Checkpoint model and ema_model are not None. Please check checkpoint's structure."
    # 'model' is default option
    if ckpt_state["model"] is None:
        logger.info(msg=f"[{device}]: Failed to load checkpoint 'model', 'ema_model' would be loaded.")
        ckpt_model = ckpt_state["ema_model"]
    else:
        if is_use_ema:
            logger.info(msg=f"[{device}]: Successfully to load checkpoint 'ema_model', using ema is True.")
            ckpt_model = ckpt_state["ema_model"]
        else:
            logger.info(msg=f"[{device}]: Successfully to load checkpoint 'model'.")
            ckpt_model = ckpt_state["model"]
    # Load the current model
    load_model_ckpt(model=model, model_ckpt=ckpt_model, is_train=is_train, is_pretrain=is_pretrain,
                    is_distributed=is_distributed, conditional=conditional)
    logger.info(msg=f"[{device}]: Successfully load model checkpoint.")
    # Train mode
    if is_train and not is_pretrain:
        # Load the previous model optimizer
        optim_weights_dict = ckpt_state["optimizer"]
        optimizer.load_state_dict(state_dict=optim_weights_dict)
        logger.info(msg=f"[{device}]: Successfully load optimizer checkpoint.")
        # Current checkpoint epoch
        start_epoch = ckpt_state["start_epoch"]
        # Next epoch
        return start_epoch + 1


def load_model_ckpt(model, model_ckpt, is_train=True, is_pretrain=False, is_distributed=False, conditional=False):
    """
    Initialize weight loading
    :param model: Model
    :param model_ckpt: Model checkpoint
    :param is_train: Whether to train mode
    :param is_pretrain: Whether to load pretrain checkpoint
    :param is_distributed:  Whether to distribute training
    :param conditional:  Whether conditional training
    :return: None
    """
    model_dict = model.state_dict()
    model_weights_dict = model_ckpt
    # Check if key contains 'module.' prefix.
    # This method is the name after training in the distribution, check the weight and delete
    if not is_train or (is_train and is_pretrain and not is_distributed):
        new_model_weights_dict = {}
        for key, value in model_weights_dict.items():
            if key.startswith("module."):
                new_key = key[len("module."):]
                new_model_weights_dict[new_key] = value
            else:
                new_model_weights_dict[key] = value
        model_weights_dict = new_model_weights_dict
        logger.info(msg="Successfully check the load weight and rename.")
    # Train mode and it is pretraining
    if is_train and is_pretrain and conditional:
        if is_distributed:
            new_model_weights_dict = {}
            # Check if key contains 'module.' prefix.
            # Create distributed training model weight
            for key, value in model_weights_dict.items():
                if not key.startswith("module."):
                    # Add 'module.'
                    new_key = "module." + key
                    new_model_weights_dict[new_key] = value
                else:
                    new_model_weights_dict[key] = value
            model_weights_dict = new_model_weights_dict
            logger.info(msg="Successfully check the load pretrain distributed weight and rename.")
            # Eliminate the impact of number of classes
            model_weights_dict["module.label_emb.weight"] = None
        else:
            # Eliminate the impact of number of classes
            model_weights_dict["label_emb.weight"] = None
    model_weights_dict = {k: v for k, v in model_weights_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
    model_dict.update(model_weights_dict)
    model.load_state_dict(state_dict=OrderedDict(model_dict))


def save_ckpt(epoch, save_name, ckpt_model, ckpt_ema_model, ckpt_optimizer, results_dir, save_model_interval,
              save_model_interval_epochs, start_model_interval, num_classes=None, conditional=None, image_size=None,
              sample=None, network=None, act=None, classes_name=None, **kwargs):
    """
    Save the model checkpoint weight files
    :param epoch: Current epoch
    :param save_name: Save the model's name
    :param ckpt_model: Model
    :param ckpt_ema_model: EMA model
    :param ckpt_optimizer: Optimizer
    :param results_dir: Results dir
    :param save_model_interval: Whether to save weight each training
    :param save_model_interval_epochs: Save model interval and save it every X epochs
    :param start_model_interval: Start epoch for saving models
    :param num_classes: Number of classes
    :param conditional: Enable conditional training
    :param image_size: Default image size
    :param sample: Sample type
    :param network: Network type
    :param act: Activation function name
    :param classes_name: All classes name
    :return: None
    """
    # Checkpoint
    ckpt_state = {
        "start_epoch": epoch, "model": ckpt_model, "ema_model": ckpt_ema_model, "optimizer": ckpt_optimizer,
        "num_classes": num_classes if conditional else 1, "classes_name": classes_name, "conditional": conditional,
        "image_size": image_size, "sample": sample, "network": network, "act": act,
    }
    # Save last checkpoint, it must be done
    last_filename = os.path.join(results_dir, f"ckpt_last.pt")
    torch.save(obj=ckpt_state, f=last_filename)
    logger.info(msg=f"Save the ckpt_last.pt")
    # If save each checkpoint, just copy the last saved checkpoint and rename it
    if save_model_interval and epoch > start_model_interval:
        if epoch % save_model_interval_epochs == 0:
            filename = os.path.join(results_dir, f"{save_name}.pt")
            shutil.copyfile(last_filename, filename)
            logger.info(msg=f"Save the {save_name}.pt")
    logger.info(msg="Finish saving the model.")


def separate_ckpt_weights(ckpt, separate_model=True, separate_ema_model=True, separate_optimizer=True):
    """
    Separate checkpoint weights
    :param ckpt: checkpoint
    :param separate_model: Whether to separate model
    :param separate_ema_model: Whether to separate ema model
    :param separate_optimizer: Whether to separate optimizer
    :return: ckpt_state
    """
    ckpt_state = ckpt.copy()
    if separate_model:
        ckpt_state["model"] = None
    if separate_ema_model:
        ckpt_state["ema_model"] = None
    if separate_optimizer:
        ckpt_state["optimizer"] = None
    return ckpt_state
