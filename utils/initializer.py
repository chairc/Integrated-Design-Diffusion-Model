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

logger = logging.getLogger(__name__)
coloredlogs.install(level="INFO")


def device_initializer():
    """
    该函数在程序第一次运行时初始化运行设备信息
    :return: cpu或cuda
    """
    logger.info(msg="Init program, it is checking the basic setting.")
    device_dict = {}
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        device = torch.device(device="cuda")
        is_init = torch.cuda.is_initialized()
        device_count = torch.cuda.device_count()
        device_name = torch.cuda.get_device_name(device=device)
        device_cap = torch.cuda.get_device_capability(device=device)
        device_prop = torch.cuda.get_device_properties(device=device)
        device_dict["is_init"] = is_init
        device_dict["device_count"] = device_count
        device_dict["device_name"] = device_name
        device_dict["device_cap"] = device_cap
        device_dict["device_prop"] = device_prop
        logger.info(msg=device_dict)
    else:
        logger.warning(msg="The device is using cpu.")
        device = torch.device(device="cpu")
    return device


def seed_initializer(seed_id=0):
    """
    初始化种子
    :param seed_id: 种子id
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
    初始化权重加载
    :param model: 模型
    :param weight_path: 权重路径
    :param device: 设备类型
    :param is_train: 是否为训练模式
    :return: None
    """
    model_dict = model.state_dict()
    model_weights_dict = torch.load(f=weight_path, map_location=device)
    # 检查键是否包含 'module.' 前缀。该方法为分布式中训练后的名称，检查权重并删除
    if not is_train:
        new_model_weights_dict = {}
        for key, value in model_weights_dict.items():
            if key.startswith('module.'):
                new_key = key[len('module.'):]
                new_model_weights_dict[new_key] = value
            else:
                new_model_weights_dict[key] = value
        model_weights_dict = new_model_weights_dict
        logger.info(msg="Successfully check the load weight and rename.")
    model_weights_dict = {k: v for k, v in model_weights_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
    model_dict.update(model_weights_dict)
    model.load_state_dict(state_dict=OrderedDict(model_dict))
