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
    @Date   : 2024/2/19 20:32
    @Author : chairc
    @Site   : https://github.com/chairc
"""
# Choice settings
# Support option
bool_choices = [True, False]
sample_choices = ["ddpm", "ddim", "plms", "dpm2", "dpmpp", "dpmpp2m", "dpmpp3m"]
network_choices = ["unet", "cspdarkunet", "unetv2", "unet-slim", "unet-cross-attn", "unet-flash-self-attn",]
optim_choices = ["adam", "adamw", "sgd"]
act_choices = ["gelu", "silu", "relu", "relu6", "lrelu"]
lr_func_choices = ["linear", "cosine", "warmup_cosine"]
image_format_choices = ["png", "jpg", "jpeg", "webp", "tif"]
noise_schedule_choices = ["linear", "cosine", "sqrt_linear", "sqrt"]
loss_func_choices = ["mse", "l1", "huber", "smooth_l1"]
generate_mode_choices = ["class", "text"]

sr_network_choices = ["srv1"]
sr_loss_func_choices = ["mse"]

autoencoder_network_choices = ["vae"]
autoencoder_loss_func_choices = ["mse_kl"]
ldm_sample_choices = ["ldm"]

image_type_choices = {"RGB": 3, "GRAY": 1}
