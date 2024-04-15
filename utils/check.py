#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Date   : 2024/4/14 17:31
    @Author : chairc
    @Site   : https://github.com/chairc
"""
import os
import logging
import coloredlogs

import torch

logger = logging.getLogger(__name__)
coloredlogs.install(level="INFO")


def check_and_create_dir(path):
    """
    Check and create not exist folder
    :param path: Create path
    :return: None
    """
    logger.info(msg=f"Check and create folder '{path}'.")
    os.makedirs(name=path, exist_ok=True)


def check_path_is_exist(path):
    """
    Check the path is existed
    :param path: Path
    :return: None
    """
    if not os.path.exists(path=path):
        raise FileNotFoundError(f"The path '{path}' does not exist.")
