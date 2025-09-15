#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Date   : 2025/9/9 10:04
    @Author : chairc
    @Site   : https://github.com/chairc
"""
import unittest
import logging
import coloredlogs
import torch

from iddm.tools.generate import Generator, init_generate_args

logger = logging.getLogger(__name__)
coloredlogs.install(level="INFO")


class TestGenerate(unittest.TestCase):
    def test_generate_fixed_tensor(self):
        """
        Test the fixed tensor generation
        """
        try:
            args = init_generate_args()
            setattr(args, "generate_name", "test_generate_fixed_tensor")
            setattr(args, "latent", True)
            setattr(args, "class_name", -1)
            setattr(args, "num_images", 6) # num_images must equal to class number when class_name is -1
            setattr(args, "weight_path", "/your/path/Defect-Diffusion-Model/weight/model.pt")
            setattr(args, "autoencoder_ckpt", "/your/path/Defect-Diffusion-Model/weight/autoencoder.pt")
            setattr(args, "result_path", "/your/path/Defect-Diffusion-Model/results/vis")
            setattr(args, "sample", "ddpm")
            num_images = getattr(args, "num_images")
            image_size = getattr(args, "image_size")
            latent = getattr(args, "latent")
            channel = 8 if latent else 3
            torch.manual_seed(0)
            gen_model = Generator(gen_args=args, deploy=False)
            image = torch.randn((num_images, channel, image_size, image_size))
            gen_model.generate(image=image)
        except Exception as e:
            logger.error(f"Error during generation: {e}")
            self.fail(f"Generation failed with exception: {e}")

    def test_generate_random_tensor(self):
        """
        Test the random tensor generation
        """
        try:
            args = init_generate_args()
            setattr(args, "generate_name", "test_generate_random_tensor")
            setattr(args, "latent", True)
            setattr(args, "class_name", -1)
            setattr(args, "num_images", 8)
            setattr(args, "weight_path", "/your/path/Defect-Diffusion-Model/weight/model.pt")
            setattr(args, "autoencoder_ckpt", "/your/path/Defect-Diffusion-Model/weight/autoencoder.pt")
            setattr(args, "result_path", "/your/path/Defect-Diffusion-Model/results/vis")
            setattr(args, "sample", "ddim")
            gen_model = Generator(gen_args=args, deploy=False)
            gen_model.generate()
        except Exception as e:
            logger.error(f"Error during generation: {e}")
            self.fail(f"Generation failed with exception: {e}")
