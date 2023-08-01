#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Date   : 2023/6/27 16:54
    @Author : chairc
    @Site   : https://github.com/chairc
"""
import argparse
import os

import torch
import unittest
import logging
import coloredlogs

from torchvision.utils import save_image
from torchsummary import summary
from matplotlib import pyplot as plt

from model.ddpm import Diffusion
from model.network import UNet
from utils.utils import get_dataset, delete_files
from utils.initializer import device_initializer
from utils.lr_scheduler import set_cosine_lr

logger = logging.getLogger(__name__)
coloredlogs.install(level="INFO")


class TestModule(unittest.TestCase):
    """
    方法测试

    1. 运行unittest测试模块
        * 运行unittest测试模块。使用python -m unittest <test_module>命令运行测试，<test_module>是测试文件的名称或相对路径
        * 例如：python -m unittest test_module.py
    2. 运行单个测试类或测试方法
        * 使用 -k 选项指定要运行的测试类或方法的名称，TestModule是要运行的测试类的名称，test_noising是要运行的测试方法的名称
        * 例如：python -m unittest -k TestModule.test_noising

    If you want to run the unittest test module, please use 'python -m unittest <test_module.py>'.
    If you want to run a single test class or test method, please use
    'python -m unittest -k <TestModule or TestClass>.<test_function_name>'.
    """

    def test_num_cases(self):
        """
        获取所有测试类名称
        :return: None
        """
        # 获取所有测试名
        test_cases = [method for method in dir(TestModule) if method.startswith('test_')]
        logger.info(test_cases)
        # 打印测试方法名称
        for method_name in test_cases:
            logger.info(method_name)

    def test_noising(self):
        """
        测试噪声
        :return: None
        """
        # 参数设置
        parser = argparse.ArgumentParser()
        parser.add_argument("--batch_size", type=int, default=1)
        parser.add_argument("--num_workers", type=int, default=2)
        # 输入图像大小
        parser.add_argument("--image_size", type=int, default=640)
        parser.add_argument("--dataset_path", type=str, default="./noising_test")

        args = parser.parse_args()
        logger.info(msg=f"Input params: {args}")

        # 开始测试
        logger.info(msg="Start noising noising_test.")
        dataset_path = args.dataset_path
        save_path = os.path.join(dataset_path, "noise")
        # 需要先清除noise文件夹下所有文件
        delete_files(path=save_path)
        dataloader = get_dataset(args=args)
        # 重新创建文件夹
        os.makedirs(name=save_path, exist_ok=True)
        # 扩散模型初始化
        diffusion = Diffusion(device="cpu")
        # 获取图像和噪声Tensor
        image = next(iter(dataloader))[0]
        time = torch.Tensor([0, 50, 125, 225, 350, 500, 675, 999]).long()

        # 给图片分别增加噪声
        noised_image, _ = diffusion.noise_images(x=image, time=time)
        # 保存噪声图片
        save_image(tensor=noised_image.add(1).mul(0.5), fp=os.path.join(save_path, "noise.jpg"))
        logger.info(msg="Finish noising noising_test.")

    def test_lr(self):
        image_size = 64
        device = device_initializer()
        net = UNet(num_classes=10, device=device, image_size=image_size)
        optimizer = torch.optim.AdamW(net.parameters(), lr=3e-4)
        lr_max = 3e-4
        lr_min = 3e-6
        max_epoch = 300
        lrs = []
        for epoch in range(max_epoch):
            set_cosine_lr(optimizer=optimizer, current_epoch=epoch, max_epoch=max_epoch, lr_min=lr_min,
                          lr_max=lr_max, warmup=True)
            logger.info(msg=f"{epoch}: {optimizer.param_groups[0]['lr']}")
            lrs.append(optimizer.param_groups[0]["lr"])
            optimizer.step()

        plt.plot(lrs)
        plt.show()

    def test_summary(self):
        """
        测试模型结构
        :return: None
        """
        image_size = 64
        device = device_initializer()
        net = UNet(num_classes=10, device=device, image_size=image_size)
        net = net.to(device)
        x = torch.randn(1, 3, image_size, image_size).to(device)
        t = x.new_tensor([500] * x.shape[0]).long().to(device)
        y = x.new_tensor([1] * x.shape[0]).long().to(device)
        print(net)
        summary(model=net, input_data=[x, t, y])


if __name__ == "__main__":
    pass
