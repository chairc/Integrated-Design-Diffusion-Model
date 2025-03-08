#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Date   : 2024/7/9 00:01
    @Author : chairc
    @Site   : https://github.com/chairc
"""
from iddm.config.choices import image_type_choices

# Temp files download path
DOWNLOAD_FILE_TEMP_PATH = "../.temp/download_files"

# Train
MASTER_ADDR = "localhost"
MASTER_PORT = "12345"
EMA_BETA = 0.995

# Some special parameter settings
# model.networks
# Image type setting (1 channel or 3 channels), you can set "RGB" or "GRAY" in the project
IMAGE_TYPE = "RGB"
IMAGE_CHANNEL = image_type_choices[IMAGE_TYPE]
TIME_CHANNEL = 256
CHANNEL_LIST = [32, 64, 128, 256, 512, 1024]

DEFAULT_IMAGE_SIZE = [64, 64]

# model.networks.sr
SR_IMAGE_TYPE = "RGB"
SR_IMAGE_CHANNEL = image_type_choices[SR_IMAGE_TYPE]

# Data processing
# ****** torchvision.transforms.Compose ******
# RandomResizedCrop
RANDOM_RESIZED_CROP_SCALE = (0.8, 1.0)
# Mean in datasets
MEAN = (0.5, 0.5, 0.5)
# Std in datasets
STD = (0.5, 0.5, 0.5)
# Sr mean in datasets
SR_MEAN = (0.485, 0.456, 0.406)
# Sr std in datasets
SR_STD = (0.229, 0.224, 0.225)

# Project
# Log level setting
LOG_LEVEL_DEFAULT = "INFO"
# config
LOG_LEVEL_CONFIG_CHOICES = "INFO"
LOG_LEVEL_CONFIG_VERSION = "INFO"
# model.modules
LOG_LEVEL_MODULES_ACT = "INFO"
LOG_LEVEL_MODULES_CONV = "INFO"
LOG_LEVEL_MODULES_MODULE = "INFO"
# model.samples
LOG_LEVEL_SAMPLES_DDPM = "INFO"
LOG_LEVEL_SAMPLES_DDIM = "INFO"
LOG_LEVEL_SAMPLES_PLMS = "INFO"
# model.trainers
LOG_LEVEL_TRAINERS_DM = "INFO"
LOG_LEVEL_TRAINERS_SR = "INFO"
# sr
LOG_LEVEL_SR_DEMO = "INFO"
LOG_LEVEL_SR_INTERFACE = "INFO"
LOG_LEVEL_SR_TRAIN = "INFO"
# Tools
LOG_LEVEL_TOOLS_DEPLOY = "INFO"
LOG_LEVEL_TOOLS_TRAIN = "INFO"
LOG_LEVEL_TOOLS_GENERATE = "INFO"
LOG_LEVEL_TOOLS_FID_PLUS = "INFO"
# utils
LOG_LEVEL_UTILS_CHECK = "INFO"
LOG_LEVEL_UTILS_CHECKPOINT = "INFO"
LOG_LEVEL_UTILS_INIT = "INFO"
LOG_LEVEL_UTILS_UTILS = "INFO"
