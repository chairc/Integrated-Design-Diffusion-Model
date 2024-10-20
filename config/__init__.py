#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Date   : 2024/3/12 14:11
    @Author : chairc
    @Site   : https://github.com/chairc
"""
from .choices import bool_choices, sample_choices, network_choices, optim_choices, act_choices, lr_func_choices, \
    image_format_choices, noise_schedule_choices
from .setting import MASTER_ADDR, MASTER_PORT, EMA_BETA, RANDOM_RESIZED_CROP_SCALE, MEAN, STD
from .version import __version__, get_versions, get_latest_version, get_old_versions, check_version_is_latest, \
    get_version_banner
