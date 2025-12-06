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

from iddm.utils.check import check_path_is_exist

logger = logging.getLogger(__name__)
coloredlogs.install(level="INFO")

# Add checkpoint caching
_checkpoint_cache = {}


def load_ckpt_with_cache(ckpt_path, device="cpu", force_reload=False):
    """
    Load checkpoint weight files with caching
    :param ckpt_path: Checkpoint path
    :param device: GPU or CPU
    :param force_reload: Whether to force reload the checkpoint
    :return: ckpt_state
    """
    cache_key = f"{ckpt_path}:{device}"
    # Return cached checkpoint if available and not forcing reload
    if not force_reload and cache_key in _checkpoint_cache:
        logger.info(f"Loading checkpoint from cache: {cache_key}")
        return _checkpoint_cache[cache_key]
    # Load checkpoint, pytorch 2.6+ default weights_only=True
    try:
        ckpt_state = torch.load(f=ckpt_path, map_location=device, weights_only=False)
        # Cache the loaded checkpoint
        _checkpoint_cache[cache_key] = ckpt_state
        logger.info(f"Checkpoint loaded and cached: {ckpt_path}")
        return ckpt_state
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint, path：{ckpt_path}, msg: {str(e)}") from e


def clear_ckpt_with_cache():
    """
    Clear checkpoint cache to free memory
    """
    global _checkpoint_cache
    _checkpoint_cache.clear()


def load_ckpt(ckpt_path, model=None, device="cpu", optimizer=None, is_train=False, is_generate=False, is_pretrain=False,
              is_resume=False, is_distributed=False, is_use_ema=False, conditional=False, ckpt_type="df",
              force_reload=False):
    """
    Load checkpoint weight files
    :param ckpt_path: Checkpoint path
    :param model: Network
    :param optimizer: Optimizer
    :param device: GPU or CPU
    :param is_train: Whether to train mode
    :param is_generate: Whether to generate mode
    :param is_pretrain: Whether to load pretrain checkpoint
    :param is_resume: Whether to resume training
    :param is_distributed:  Whether to distribute training
    :param is_use_ema:  Whether to use ema model or model
    :param conditional:  Whether conditional training
    :param ckpt_type: Type of checkpoint
    :param force_reload: Whether to force reload the checkpoint
    :return: start_epoch + 1
    """
    # ============================= Check and load checkpoint ============================= #
    logger.info(f"Current load model status: [train: {is_train}, generate: {is_generate}, pretrain: {is_pretrain}, "
                f"resume: {is_resume}, distributed: {is_distributed}, use_ema: {is_use_ema}, "
                f"conditional: {conditional}, type: {ckpt_type}]")
    # Check path
    check_path_is_exist(path=ckpt_path)
    # Load checkpoint, pytorch 2.6+ default weights_only=True
    ckpt_state = load_ckpt_with_cache(ckpt_path=ckpt_path, device=device, force_reload=force_reload)

    # ============================= Get the result info ============================= #
    # Load the model best score and mode
    model_score, mode = [], None
    # Super resolution checkpoint
    if ckpt_type == "sr":
        model_score.append(ckpt_state.get("best_ssim", 0.0))
        model_score.append(ckpt_state.get("best_psnr", 0.0))
    # Autoencoder checkpoint
    elif ckpt_type == "autoencoder":
        model_score.append(ckpt_state.get("best_score", 0.0))
    # Diffusion model checkpoint
    else:
        mode = ckpt_state.get("mode", "class")

    # ============================= Get pretrain status ============================= #
    # Pretrain checkpoint
    if is_pretrain:
        logger.info(msg=f"[{device}]: Load pretrain checkpoint[{ckpt_type}], path is '{ckpt_path}'.")
    else:
        logger.info(msg=f"[{device}]: Load checkpoint[{ckpt_type}], path is '{ckpt_path}'.")

    # ============================= Load model or ema model ============================= #
    # Check checkpoint's structure
    has_model = ckpt_state.get("model") is not None
    has_ema_model = ckpt_state.get("ema_model") is not None
    assert has_model or has_ema_model, \
        f"Error!! Checkpoint model and ema_model are not None. Please check checkpoint[{ckpt_type}]'s structure."
    # 'model' is default option
    if is_use_ema:
        if has_ema_model:
            ckpt_model = ckpt_state["ema_model"]
            logger.info(msg=f"[{device}]: Successfully to load checkpoint[{ckpt_type}] 'ema_model', using ema is True.")
        else:
            ckpt_model = ckpt_state["model"]
            logger.info(msg=f"[{device}]: Successfully to load checkpoint[{ckpt_type}] 'model'.")
    else:
        if has_model:
            ckpt_model = ckpt_state["model"]
            logger.info(msg=f"[{device}]: Successfully to load checkpoint[{ckpt_type}] 'model'.")
        else:
            ckpt_model = ckpt_state["ema_model"]
            logger.info(
                msg=f"[{device}]: Successfully to load checkpoint[{ckpt_type}] 'ema_model', using ema is False.")

    # ============================= Load model params ============================= #
    # Load the current model
    if model is not None:
        load_model_ckpt(model=model, model_ckpt=ckpt_model, is_train=is_train, is_generate=is_generate,
                        is_pretrain=is_pretrain, is_distributed=is_distributed, conditional=conditional, mode=mode)
        logger.info(msg=f"[{device}]: Successfully load model's checkpoint[{ckpt_type}].")

    # ============================= Resume training ============================= #
    # Train mode, resume training
    if is_train and is_resume:
        # Load the previous model optimizer
        if optimizer is None:
            raise ValueError("Optimizer is None, please set the optimizer.")
        optim_weights_dict = ckpt_state.get("optimizer")
        if optim_weights_dict is None:
            raise ValueError(f"Error!! Checkpoint[{ckpt_type}] optimizer is None, unable to return to training status, "
                             f"please check checkpoint's structure.")
        try:
            optimizer.load_state_dict(state_dict=optim_weights_dict)
            logger.info(msg=f"[{device}]: Successfully load optimizer checkpoint[{ckpt_type}].")
        except Exception as e:
            raise RuntimeError(f"Failed to load optimizer checkpoint[{ckpt_type}], msg: {str(e)}") from e
        # Current checkpoint epoch
        start_epoch = ckpt_state.get("start_epoch", -1)
        # Next epoch
        next_epoch = start_epoch + 1
        logger.info(f"[{device}]: Resume training, current epoch：{start_epoch}, next epoch：{next_epoch}")
        return next_epoch, model_score
    return None


def load_model_ckpt(model, model_ckpt, is_train=False, is_generate=False, is_pretrain=False, is_distributed=False,
                    conditional=False, mode=None):
    """
    Initialize weight loading
    :param model: Model
    :param model_ckpt: Model checkpoint
    :param is_train: Whether to train mode
    :param is_generate: Whether to generate mode
    :param is_pretrain: Whether to load pretrain checkpoint
    :param is_distributed:  Whether to distribute training
    :param conditional:  Whether conditional training for diffusion model
    :param mode: Mode type for diffusion model
    :return: None
    """

    def adjust_module_prefix(weights, add_prefix):
        """
        Uniformly handle the 'module.' prefix in the weight key name
        :param weights: Original weights dictionary
        :param add_prefix: Whether to add 'module.' prefix（True/False）
        :return: Adjusted weights dictionary
        """
        adjusted = {}
        for k, v in weights.items():
            if add_prefix:
                # If you need to add a prefix but don't have it at the moment, add it
                if not k.startswith("module."):
                    adjusted[f"module.{k}"] = v
                else:
                    adjusted[k] = v
            else:
                # If the prefix needs to be removed but is currently present, strip it
                if k.startswith("module."):
                    adjusted[k[len("module."):]] = v
                else:
                    adjusted[k] = v
        return adjusted

    logger.info(msg=f"Current load model status: [train: {is_train}, pretrain: {is_pretrain}, "
                    f"distributed: {is_distributed}, conditional: {conditional}, mode: {mode}]")
    # Load model state dict
    model_dict = model.state_dict()
    model_weights_dict = model_ckpt
    # Check if key contains 'module.' prefix.
    # This method is the name after training in the distribution, check the weight and delete
    if is_generate or (is_train and is_pretrain and not is_distributed):
        model_weights_dict = adjust_module_prefix(weights=model_weights_dict, add_prefix=False)
        logger.info(msg="Successfully check the load weight and rename.")
    # Distinguish between distributed and non-distributed weight keys
    prefix = "module." if is_distributed else ""
    # Train mode and it is pretraining and conditional for diffusion model
    if is_train and is_pretrain and conditional:
        if is_distributed:
            model_weights_dict = adjust_module_prefix(weights=model_weights_dict, add_prefix=True)
            logger.info(msg="Successfully check the load pretrain distributed weight and rename.")
            # Eliminate the impact of number of classes
        if mode == "class":
            # Compatibility with new and old model categories mismatch
            target_keys = [
                f"{prefix}label_emb.weight",
                f"{prefix}condition_adapter.label_emb.weight"
            ]
            # Clear target weights
            for key in target_keys:
                if key in model_weights_dict:
                    model_weights_dict[key] = None
                    logger.debug(f"Incompatible weights have been cleared: {key}")

    if is_generate and conditional and mode == "class":
        old_key = f"{prefix}label_emb.weight"
        new_key = f"{prefix}condition_adapter.label_emb.weight"
        if old_key in model_weights_dict:
            old_value = model_weights_dict[old_key]
            del model_weights_dict[old_key]
            model_weights_dict[new_key] = old_value

    # Filter weights with mismatched shapes and update the model status dictionary
    model_weights_dict = {
        k: v for k, v in model_weights_dict.items() if
        k in model_dict and np.shape(model_dict[k]) == np.shape(v)
    }
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
    # Check mode type
    # Super resolution
    if kwargs.get("is_sr", False):
        ckpt_state = {
            "start_epoch": epoch, "model": ckpt_model, "ema_model": ckpt_ema_model,
            "optimizer": ckpt_optimizer, "image_size": image_size, "network": network, "act": act, "ssim": -1,
            "psnr": -1, "best_ssim": -1, "best_psnr": -1
        }
        ssim, psnr = kwargs.get("ssim"), kwargs.get("psnr")
        best_ssim, best_psnr = kwargs.get("best_ssim"), kwargs.get("best_psnr")
        ckpt_state["ssim"] = ssim
        ckpt_state["psnr"] = psnr
        ckpt_state["best_ssim"] = best_ssim
        ckpt_state["best_psnr"] = best_psnr
        message_info = f"current ssim is {ssim}, psnr is {psnr}, best ssim is {best_ssim}, best psnr is {best_psnr}"
    # Autoencoder
    elif kwargs.get("is_autoencoder", False):
        ckpt_state = {
            "start_epoch": epoch, "model": ckpt_model, "ema_model": ckpt_ema_model, "optimizer": ckpt_optimizer,
            "image_size": image_size, "network": network, "act": act, "score": -1, "best_score": -1,
        }
        score = kwargs.get("score")
        best_score = kwargs.get("best_score")
        latent_channel = kwargs.get("latent_channel", 4)
        ckpt_state["score"] = score
        ckpt_state["best_score"] = best_score
        ckpt_state["latent_channel"] = latent_channel
        message_info = f"current score is {score}, best score is {best_score}"
    # Diffusion model
    else:
        mode = kwargs.get("mode", None)
        ckpt_state = {
            "start_epoch": epoch, "model": ckpt_model, "ema_model": ckpt_ema_model, "optimizer": ckpt_optimizer,
            "num_classes": num_classes if conditional else 1, "classes_name": classes_name, "conditional": conditional,
            "image_size": image_size, "sample": sample, "network": network, "act": act, "mode": mode
        }
        message_info = ""

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

    # Check is the best model and save it
    if kwargs.get("is_best", False):
        best_filename = os.path.join(results_dir, f"ckpt_best.pt")
        shutil.copyfile(last_filename, best_filename)
        logger.info(msg=f"Save the ckpt_best.pt, {message_info}")
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
