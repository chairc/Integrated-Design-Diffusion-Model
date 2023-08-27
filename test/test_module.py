#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Date   : 2023/6/27 16:54
    @Author : chairc
    @Site   : https://github.com/chairc
"""
import argparse
import json
import os
import socket

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
    Test Module

    1. Run the unittest test module
        * If you want to run the unittest test module, please use 'python -m unittest <test_module.py>',
        <test module> is the name or relative path of the test file
        * e.g: python -m unittest test_module.py
    2. Run a single test class or test method
        * Use the -k option to specify the name of the test class or method to run,
        where TestModule is the name of the test class to run and test_noising is the name of the test method to run.
        * e.g: python -m unittest -k TestModule.test_noising

    """

    def test_num_cases(self):
        """
        Get all test class names
        :return: None
        """
        # Get all test class names
        test_cases = [method for method in dir(TestModule) if method.startswith('test_')]
        logger.info(test_cases)
        # Print all test class names
        for method_name in test_cases:
            logger.info(method_name)

    def test_noising(self):
        """
        Test noising
        :return: None
        """
        # Parameter settings
        parser = argparse.ArgumentParser()
        parser.add_argument("--batch_size", type=int, default=1)
        parser.add_argument("--num_workers", type=int, default=2)
        # Input image size
        parser.add_argument("--image_size", type=int, default=640)
        parser.add_argument("--dataset_path", type=str, default="./noising_test")

        args = parser.parse_args()
        logger.info(msg=f"Input params: {args}")

        # Start test
        logger.info(msg="Start noising noising_test.")
        dataset_path = args.dataset_path
        save_path = os.path.join(dataset_path, "noise")
        # You need to clear all files under the 'noise' folder first
        delete_files(path=save_path)
        dataloader = get_dataset(args=args)
        # Recreate the folder
        os.makedirs(name=save_path, exist_ok=True)
        # Diffusion model initialization
        diffusion = Diffusion(device="cpu")
        # Get image and noise tensor
        image = next(iter(dataloader))[0]
        time = torch.Tensor([0, 50, 125, 225, 350, 500, 675, 999]).long()

        # Add noise to the image
        noised_image, _ = diffusion.noise_images(x=image, time=time)
        # Save noise images
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
        Test model structure
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

    def test_send_message(self):
        """
        Test local send message to deploy.py
        :return: None
        """
        test_json = {"conditional": True, "sample": "ddpm", "image_size": 64, "num_images": 2,
                     "weight_path": "/your/test/model/path/test.pt",
                     "result_path": "/your/results/deploy",
                     "num_classes": 6, "class_name": 1, "cfg_scale": 3}
        logger.info(msg=f"Test json: {test_json}")
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # host = "127.0.1.1"
        # host = "192.168.16.1"
        host = socket.gethostname()
        client_socket.bind((host, 12346))
        client_socket.connect((host, 12345))
        msg = json.dumps(test_json)
        client_socket.send(msg.encode("utf-8"))
        client_socket.send("-iccv-over".encode("utf-8"))
        client_socket.close()
        logger.info(msg="Send message successfully!")


if __name__ == "__main__":
    pass
