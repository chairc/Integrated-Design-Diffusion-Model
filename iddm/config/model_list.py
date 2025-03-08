#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Date   : 2024/11/18 19:57
    @Author : chairc
    @Site   : https://github.com/chairc
"""

pretrain_model_choices = {
    "df": {
        "default": {
            "unet": {
                "conditional": {
                    "64": "https://github.com/chairc/Integrated-Design-Diffusion-Model/releases/download/v1.1.5/cifar10-64-weight.pt",
                    "120": "https://github.com/chairc/Integrated-Design-Diffusion-Model/releases/download/v1.1.5/celebahq-120-weight.pt",
                },
                "unconditional": {
                    "64": "",
                    "120": "",
                },
            },
            "unetv2": {
                "conditional": {
                    "64": "",
                    "120": "",
                },
                "unconditional": {
                    "64": "",
                    "120": "",
                },
            },
            "cspdarkunet": {
                "conditional": {
                    "64": "",
                    "120": "",
                },
                "unconditional": {
                    "64": "",
                    "120": "",
                },
            }
        },
        "exp": {
            "unet": {
                "gelu": {
                    "64": {
                        "neu-cls": "https://github.com/chairc/Integrated-Design-Diffusion-Model/releases/download/v1.1.7/neu-cls-64-weight.pt",
                        "cifar10": "https://github.com/chairc/Integrated-Design-Diffusion-Model/releases/download/v1.1.5/cifar10-64-weight.pt",
                        "animate-face": "https://github.com/chairc/Integrated-Design-Diffusion-Model/releases/download/v1.1.5/animate-face-64-weight.pt"
                    },
                    "120": {
                        "neu": "https://github.com/chairc/Integrated-Design-Diffusion-Model/releases/download/v1.1.5/neu-120-weight.pt",
                        "animate-ganyu": "https://github.com/chairc/Integrated-Design-Diffusion-Model/releases/download/v1.1.5/animate-ganyu-120-weight.pt",
                        "celebahq": "https://github.com/chairc/Integrated-Design-Diffusion-Model/releases/download/v1.1.5/celebahq-120-weight.pt",
                    }
                }
            },
        }
    },
    "sr": {
        "srv1": {
            "gelu": {
                "64": "",
                "120": "",
            },
            "silu": {
                "64": "",
                "120": "",
            },
            "relu": {
                "64": "",
                "120": "",
            },
            "relu6": {
                "64": "",
                "120": "",
            },
            "lrelu": {
                "64": "",
                "120": "",
            },
        }
    }
}

if __name__ == "__main__":
    pass