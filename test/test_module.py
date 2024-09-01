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

from config.choices import parse_image_size_type
from utils.dataset import get_dataset, set_resize_images_size
from utils.utils import delete_files
from utils.initializer import device_initializer, network_initializer, sample_initializer, generate_initializer
from utils.lr_scheduler import set_cosine_lr
from utils.check import check_is_nan, check_image_size
from utils.checkpoint import separate_ckpt_weights

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
        # Start test
        logger.info(msg="Start noising noising_test.")
        image_size = 64
        batch_size = 1
        num_workers = 2
        dataset_path = "./noising_test"
        save_path = os.path.join(dataset_path, "noise")
        # You need to clear all files under the 'noise' folder first
        delete_files(path=save_path)
        dataloader = get_dataset(image_size=image_size, dataset_path=dataset_path, batch_size=batch_size,
                                 num_workers=num_workers)
        # Recreate the folder
        os.makedirs(name=save_path, exist_ok=True)
        # Diffusion model initialization
        diffusion = sample_initializer(sample="ddpm", image_size=image_size, device="cpu")
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
        Net = network_initializer(network="unet", device=device)
        net = Net(num_classes=10, device=device, image_size=image_size)
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
        # Select model
        # Option: unet/cspdarkunet
        model = "cspdarkunet"
        device = device_initializer()
        x = torch.randn(1, 3, image_size, image_size).to(device)
        t = x.new_tensor([500] * x.shape[0]).long().to(device)
        y = x.new_tensor([1] * x.shape[0]).long().to(device)
        Net = network_initializer(network=model, device=device)
        net = Net(num_classes=10, device=device, image_size=image_size)
        net = net.to(device)
        print(net)
        summary(model=net, input_data=[x, t, y])

    def test_send_message(self):
        """
        Test local send message to deploy.py
        :return: None
        """
        test_json = {"conditional": True, "sample": "ddpm", "image_size": 64, "num_images": 2, "act": "gelu",
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

    def test_separate_ckpt_weights(self):
        """
        test separate checkpoint weights
        :return: None
        """
        # Checkpoint path
        ckpt_root_path = "/your/checkpoint/root/path"
        ckpt_name = "your_checkpoint.pt"
        ckpt_path = os.path.join(ckpt_root_path, ckpt_name)
        # Load model
        ckpt = torch.load(f=ckpt_path, map_location="cpu")
        print(len(ckpt["model"]), len(ckpt["ema_model"]), len(ckpt["optimizer"]))
        ckpt_model = separate_ckpt_weights(ckpt=ckpt, separate_model=False)
        ckpt_ema_model = separate_ckpt_weights(ckpt=ckpt, separate_ema_model=False)
        ckpt_optimizer = separate_ckpt_weights(ckpt=ckpt, separate_optimizer=False)
        print(len(ckpt_model["model"]), ckpt_model["ema_model"], ckpt_model["optimizer"])
        print(ckpt_ema_model["model"], len(ckpt_ema_model["ema_model"]), ckpt_ema_model["optimizer"])
        print(ckpt_optimizer["model"], ckpt_optimizer["ema_model"], len(ckpt_optimizer["optimizer"]))
        torch.save(obj=ckpt_model, f=os.path.join(ckpt_root_path, "ckpt_model.pt"))
        logger.info(msg="Save ckpt_model successfully!")
        torch.save(obj=ckpt_ema_model, f=os.path.join(ckpt_root_path, "ckpt_ema_model.pt"))
        logger.info(msg="Save ckpt_ema_model successfully!")
        torch.save(obj=ckpt_optimizer, f=os.path.join(ckpt_root_path, "ckpt_optimizer.pt"))
        logger.info(msg="Save ckpt_optimizer successfully!")

    def test_ckpt_parameters(self):
        """
        Test checkpoint parameters validity
        :return: None
        """
        parser = argparse.ArgumentParser()
        parser.add_argument("--conditional", type=bool, default=True)
        parser.add_argument("--image_size", type=int, default=128)
        parser.add_argument("--sample", type=str, default="ddpm")
        parser.add_argument("--network", type=str, default="cspdarkunet")
        parser.add_argument("--act", type=str, default="gelu")
        parser.add_argument("--num_classes", type=int, default=10)
        args = parser.parse_args()
        device = device_initializer()
        ckpt_path = "/your/test/model/path/test.pt"
        results = generate_initializer(ckpt_path=ckpt_path, args=args, device=device)
        logger.info(
            msg=f"Parser parameters: {(args.conditional, args.sample, args.network, args.image_size, args.num_classes, args.act)}")
        logger.info(msg=f"Return parameters: {results}")

    def test_pre_ckpt_add_parameters(self):
        conditional = True
        image_size = 64
        sample = "ddim"
        network = "unet"
        act = "gelu"
        num_classes = 10
        classes_name = None
        root_ckpt_path = "/your/test/model/path"
        ckpt_name = "test.pt"
        device = device_initializer()
        ckpt_state = torch.load(f=os.path.join(root_ckpt_path, ckpt_name), map_location=device)
        new_ckpt_state = {
            "start_epoch": ckpt_state["start_epoch"], "model": ckpt_state["model"],
            "ema_model": ckpt_state["ema_model"], "optimizer": ckpt_state["optimizer"],
            "num_classes": num_classes if conditional else 1, "classes_name": classes_name, "conditional": conditional,
            "image_size": image_size, "sample": sample, "network": network, "act": act,
        }
        torch.save(obj=new_ckpt_state, f=os.path.join(root_ckpt_path, "new_ckpt.pt"))
        logger.info(
            msg=f"Parser parameters: {(ckpt_state['start_epoch'], len(ckpt_state['model']), len(ckpt_state['ema_model']), len(ckpt_state['optimizer']))}")

    def test_nan(self):
        """
        Test nan
        :return: None
        """
        nan = torch.tensor(float("nan"))
        correct_tensor = torch.tensor(1)
        correct_tensor_list = torch.tensor([1, 0.5, 4])
        assert check_is_nan(tensor=nan)
        assert not check_is_nan(tensor=correct_tensor)
        assert not check_is_nan(tensor=correct_tensor_list)

    def test_resize_images_size(self):
        """
        Test resize images size
        :return: None
        """
        divisor = 4
        image_size1 = 64
        image_size2 = [64, 32]
        image_size3 = (64, 32)
        res1 = set_resize_images_size(image_size=image_size1, divisor=divisor)
        res2 = set_resize_images_size(image_size=image_size2, divisor=divisor)
        res3 = set_resize_images_size(image_size=image_size3, divisor=divisor)
        logger.info(msg=res1)
        logger.info(msg=res2)
        logger.info(msg=res3)
        assert res1 == 80
        assert res2 == [80, 40]
        assert res3 == (80, 40)

    def test_parse_image_size_type(self):
        """
        Test parse image size type
        """
        image_size1 = "64"
        image_size2 = "[64,32]"
        image_size3 = "(64,32)"
        res1 = parse_image_size_type(image_size1)
        res2 = parse_image_size_type(image_size2)
        res3 = parse_image_size_type(image_size3)
        logger.info(msg=res1)
        logger.info(msg=res2)
        logger.info(msg=res3)
        assert res1 == 64
        assert res2 == [64, 32]
        assert res3 == (64, 32)

    def test_check_image_size(self):
        """
        Test check image size
        """
        image_size1 = 64
        image_size2 = [64, 32]
        image_size3 = (64, 32)
        image_size4 = None
        res1 = check_image_size(image_size=image_size1)
        res2 = check_image_size(image_size=image_size2)
        res3 = check_image_size(image_size=image_size3)
        res4 = check_image_size(image_size=image_size4)
        logger.info(msg=res1)
        logger.info(msg=res2)
        logger.info(msg=res3)
        logger.info(msg=res4)
        assert res1 == [64, 64]
        assert res2 == [64, 32]
        assert res3 == [64, 32]
        assert res4 == [64, 64]


if __name__ == "__main__":
    pass
