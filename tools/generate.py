#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Date   : 2023/6/20 22:33
    @Author : chairc
    @Site   : https://github.com/chairc
"""
import os
import sys
import argparse

import torch
import logging
import coloredlogs

sys.path.append(os.path.dirname(sys.path[0]))
from model.ddpm import Diffusion
from model.network import UNet
from utils.initializer import device_initializer
from utils.utils import plot_images, save_images

logger = logging.getLogger(__name__)
coloredlogs.install(level="INFO")


def generate(args):
    logger.info(msg="Start generation.")
    # 是否启用条件生成
    conditional = args.conditional
    # 生成名称
    generate_name = args.generate_name
    # 图片大小
    image_size = args.image_size
    # 图片个数
    num_images = args.num_images
    # 权重路径
    weight_path = args.weight_path
    # 保存路径
    result_path = args.result_path
    # 设备初始化
    device = device_initializer()
    # 模型初始化
    if conditional:
        # 类别个数
        num_classes = args.num_classes
        # 生成的类别名称
        class_name = args.class_name
        # classifier-free guidance插值权重
        cfg_scale = args.cfg_scale
        model = UNet(num_classes=num_classes, device=device, image_size=image_size).to(device)
        # 加载权重路径
        weight = torch.load(f=weight_path)
        model.load_state_dict(state_dict=weight)
        diffusion = Diffusion(img_size=image_size, device=device)
        y = torch.Tensor([class_name] * num_images).long().to(device)
        x = diffusion.sample(model=model, n=num_images, labels=y, cfg_scale=cfg_scale)
    else:
        model = UNet(device=device, image_size=image_size).to(device)
        # 加载权重路径
        weight = torch.load(f=weight_path)
        model.load_state_dict(state_dict=weight)
        diffusion = Diffusion(img_size=image_size, device=device)
        x = diffusion.sample(model=model, n=num_images)
    # 如果不存在路径信息则只展示；存在则保存到指定路径并展示
    if result_path == "" or result_path is None:
        plot_images(images=x)
    else:
        save_images(images=x, path=os.path.join(result_path, f"{generate_name}.jpg"))
        plot_images(images=x)
    logger.info(msg="Finish generation.")


if __name__ == "__main__":
    # 生成模型参数
    parser = argparse.ArgumentParser()
    # 生成名称
    parser.add_argument("--generate_name", type=str, default="df")
    # 输入图像大小
    parser.add_argument("--image_size", type=int, default=64)
    # 生成图片个数
    parser.add_argument("--num_images", type=int, default=8)
    # 模型路径
    parser.add_argument("--weight_path", type=str, default="/your/path/Defect-Diffusion-Model/weight/model.pt")
    # 保存路径
    parser.add_argument("--result_path", type=str, default="/your/path/Defect-Diffusion-Model/results/vis")
    # 开启条件生成，若使用False则不需要设置该参数之后的参数
    parser.add_argument("--conditional", type=bool, default=False)

    # ==========================开启条件生成分界线==========================
    # 类别个数
    parser.add_argument("--num_classes", type=int, default=10)
    # 类别名称
    parser.add_argument("--class_name", type=int, default=0)
    # classifier-free guidance插值权重，用户更好生成模型效果
    parser.add_argument("--cfg_scale", type=int, default=3)

    args = parser.parse_args()
    generate(args)
