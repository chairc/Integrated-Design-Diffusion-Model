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
    @Date   : 2023/11/8 22:43
    @Author : chairc
    @Site   : https://github.com/chairc
"""
# Diffusion model network
from .base import BaseNet
from .cspdarkunet import CSPDarkUnet
from .unet import UNet
from .unet_cross_attn import UNetCrossAttn
from .unet_flash_self_attn import UNetFlashSelfAttn
from .unetv2 import UNetV2
from .unet_slim import UNetSlim

# Super resolution network
from .sr.srv1 import SRv1
