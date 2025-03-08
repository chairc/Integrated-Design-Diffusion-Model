#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Date   : 2024/3/12 14:11
    @Author : chairc
    @Site   : https://github.com/chairc
"""
from .choices import bool_choices, sample_choices, network_choices, optim_choices, act_choices, lr_func_choices, \
    image_format_choices, noise_schedule_choices, loss_func_choices, image_type_choices, sr_network_choices, \
    sr_loss_func_choices, parse_image_size_type
from .setting import MASTER_ADDR, MASTER_PORT, EMA_BETA, RANDOM_RESIZED_CROP_SCALE, MEAN, STD, \
    DOWNLOAD_FILE_TEMP_PATH, IMAGE_TYPE, IMAGE_CHANNEL, TIME_CHANNEL, CHANNEL_LIST, DEFAULT_IMAGE_SIZE, \
    SR_IMAGE_CHANNEL, SR_IMAGE_TYPE, SR_STD, SR_MEAN
from .version import __version__, get_versions, get_latest_version, get_old_versions, check_version_is_latest, \
    get_version_banner
from .model_list import pretrain_model_choices
