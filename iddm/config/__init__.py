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
    @Date   : 2024/3/12 14:11
    @Author : chairc
    @Site   : https://github.com/chairc
"""
from .choices import bool_choices, sample_choices, network_choices, optim_choices, act_choices, lr_func_choices, \
    image_format_choices, noise_schedule_choices, loss_func_choices, image_type_choices, sr_network_choices, \
    sr_loss_func_choices, generate_mode_choices
from .setting import MASTER_ADDR, MASTER_PORT, EMA_BETA, RANDOM_RESIZED_CROP_SCALE, MEAN, STD, \
    DOWNLOAD_FILE_TEMP_PATH, IMAGE_TYPE, IMAGE_CHANNEL, TIME_CHANNEL, CHANNEL_LIST, DEFAULT_IMAGE_SIZE, \
    SR_IMAGE_CHANNEL, SR_IMAGE_TYPE, SR_STD, SR_MEAN
from .version import __version__, get_versions, get_latest_version, get_old_versions, check_version_is_latest, \
    get_version_banner
from .model_list import pretrain_model_choices
