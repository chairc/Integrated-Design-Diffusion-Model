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
    @Date   : 2024/9/18 13:09
    @Author : chairc
    @Site   : https://github.com/chairc
"""
import os
import sys
import copy
import logging
import time

import coloredlogs
import torch

from torch import nn as nn
from torch.amp import autocast
from tqdm import tqdm

sys.path.append(os.path.dirname(sys.path[0]))
from iddm.config.setting import EMA_BETA
from iddm.model.modules.ema import EMA
from iddm.model.trainers.base import Trainer
from iddm.utils.initializer import seed_initializer, sr_network_initializer, optimizer_initializer, amp_initializer, \
    loss_initializer
from iddm.utils.utils import save_images, check_and_create_dir
from iddm.utils.checkpoint import load_ckpt, save_ckpt
from iddm.utils.metrics import compute_psnr, compute_ssim
from iddm.sr.interface import post_image
from iddm.sr.dataset import get_sr_dataset

logger = logging.getLogger(__name__)
coloredlogs.install(level="INFO")


class SRTrainer(Trainer):
    """
    Super resolution trainer
    """

    def __init__(self, **kwargs):
        """
        Initialize super resolution trainer
        :param kwargs: Parameters of trainer
        """
        super(SRTrainer, self).__init__(**kwargs)
        # Can be set input params
        # Run name
        self.run_name = self.check_args_and_kwargs(kwarg="run_name", default="sr")
        # Datasets
        self.train_dataset_path = self.check_args_and_kwargs(kwarg="train_dataset_path", default="")
        self.val_dataset_path = self.check_args_and_kwargs(kwarg="val_dataset_path", default="")
        # Evaluate quickly
        self.quick_eval = self.check_args_and_kwargs(kwarg="quick_eval", default=False)

        # Default params
        self.train_dataloader = None
        self.val_dataloader = None
        self.len_train_dataloader = None
        self.len_val_dataloader = None
        self.save_val_vis_dir = None
        self.best_ssim = 0
        self.best_psnr = 0
        self.avg_val_loss = 0
        self.avg_ssim = 0
        self.avg_psnr = 0

    def before_train(self):
        """
        Before training super resolution model method
        """
        # =================================Before training=================================
        logger.info(msg=f"[{self.rank}]: Start super resolution model training")
        logger.info(msg=f"[{self.rank}]: Input params: {self.args}")
        # Step1: Set path and create log
        # Create data logging path
        self.init_trainer_results_dir_and_log()

        # Step2: Get the parameters of the initializer and args
        # Initialize the seed
        seed_initializer(seed_id=self.seed)
        # Initialize and save the model identification bit
        self.init_trainer_distributed()

        # =================================About model initializer=================================
        # Step3: Init model
        # Network
        sr_model = sr_network_initializer(network=self.network, device=self.device)
        # Model
        self.model = sr_model(act=self.act).to(self.device)
        # Distributed training
        if self.distributed:
            self.model = nn.parallel.DistributedDataParallel(module=self.model, device_ids=[self.device],
                                                             find_unused_parameters=True)
        # Model optimizer
        self.optimizer = optimizer_initializer(model=self.model, optim=self.optim, init_lr=self.init_lr,
                                               device=self.device)
        # Resume training
        if self.resume:
            ckpt_path = None
            # Determine which checkpoint to load
            # 'start_epoch' is correct
            if self.start_epoch is not None:
                ckpt_path = os.path.join(self.results_dir, f"ckpt_{str(self.start_epoch - 1).zfill(3)}.pt")
            # Parameter 'ckpt_path' is None in the train mode
            if ckpt_path is None:
                ckpt_path = os.path.join(self.results_dir, "ckpt_last.pt")
            # Get model state
            self.start_epoch, model_score = load_ckpt(ckpt_path=ckpt_path, model=self.model, device=self.device,
                                                      optimizer=self.optimizer, is_train=True,
                                                      is_distributed=self.distributed, is_resume=self.resume,
                                                      ckpt_type="sr")
            # Get best ssim and psnr
            self.best_ssim, self.best_psnr = model_score[0], model_score[1]
            logger.info(msg=f"[{self.device}]: Successfully load resume model checkpoint.")
        else:
            # Pretrain mode
            if self.pretrain:
                load_ckpt(ckpt_path=self.pretrain_path, model=self.model, device=self.device, is_train=True,
                          is_pretrain=self.pretrain, is_distributed=self.distributed, ckpt_type="sr")
                logger.info(msg=f"[{self.device}]: Successfully load pretrain model checkpoint.")
            # Init
            self.start_epoch, self.best_ssim, self.best_psnr = 0, 0, 0
        logger.info(msg=f"[{self.device}]: The start epoch is {self.start_epoch}, best ssim is {self.best_ssim}, "
                        f"best psnr is {self.best_psnr}.")
        # Set harf-precision
        self.scaler = amp_initializer(amp=self.amp, device=self.device)
        # Loss function
        self.loss_func = loss_initializer(loss_name=self.loss_name, device=self.device)
        # Exponential Moving Average (EMA) may not be as dominant for single class as for multi class
        self.ema = EMA(beta=EMA_BETA)
        # EMA model
        self.ema_model = copy.deepcopy(self.model).eval().requires_grad_(False)

        # =================================About data=================================
        # Step4: Set data
        # Dataloader
        self.train_dataloader = get_sr_dataset(image_size=self.image_size, dataset_path=self.train_dataset_path,
                                               batch_size=self.batch_size,
                                               num_workers=self.num_workers, distributed=self.distributed)
        # Quick eval batch size
        val_batch_size = self.batch_size if self.quick_eval else 1
        self.val_dataloader = get_sr_dataset(image_size=self.image_size, dataset_path=self.val_dataset_path,
                                             batch_size=val_batch_size,
                                             num_workers=self.num_workers, distributed=self.distributed)
        # Number of dataset batches in the dataloader
        self.len_train_dataloader = len(self.train_dataloader)
        self.len_val_dataloader = len(self.val_dataloader)

    def before_iter(self):
        """
        Before training one iter super resolution model method
        """
        super().before_iter()
        # Create vis dir
        self.save_val_vis_dir = os.path.join(self.results_vis_dir, str(self.epoch))
        check_and_create_dir(self.save_val_vis_dir)

    def train_in_iter(self):
        """
        Train in one iter super resolution model method
        """
        # Initialize images and labels
        train_loss_list, val_loss_list, ssim_list, psnr_list = [], [], [], []
        # Train
        self.model.train()
        logger.info(msg="Start train mode.")
        train_pbar = tqdm(self.train_dataloader)
        for i, (lr_images, hr_images) in enumerate(train_pbar):
            # The images are all resized in train dataloader
            lr_images = lr_images.to(self.device)
            hr_images = hr_images.to(self.device)
            # Enable Automatic mixed precision training
            # Automatic mixed precision training
            # Note: Pytorch version must > 1.10
            with autocast("cuda", enabled=self.amp):
                output = self.model(lr_images)
                # To calculate the MSE loss
                # You need to use the standard normal distribution of x at time t and the predicted noise
                train_loss = self.loss_func(output, hr_images)
            # The optimizer clears the gradient of the model parameters
            self.optimizer.zero_grad()
            # Update loss and optimizer
            # Fp16 + Fp32
            self.scaler.scale(train_loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            # EMA
            self.ema.step_ema(ema_model=self.ema_model, model=self.model)

            # TensorBoard logging
            train_pbar.set_postfix(MSE=train_loss.item())
            self.tb_logger.add_scalar(tag=f"[{self.device}]: Train loss({self.loss_func})",
                                      scalar_value=train_loss.item(),
                                      global_step=self.epoch * self.len_train_dataloader + i)
            train_loss_list.append(train_loss.item())
        # Loss per epoch
        self.tb_logger.add_scalar(tag=f"[{self.device}]: Train loss",
                                  scalar_value=sum(train_loss_list) / len(train_loss_list),
                                  global_step=self.epoch)
        logger.info(msg="Finish train mode.")

        # Val
        self.model.eval()
        logger.info(msg="Start val mode.")
        val_pbar = tqdm(self.val_dataloader)
        for i, (lr_images, hr_images) in enumerate(val_pbar):
            # The images are all resized in val dataloader
            lr_images = lr_images.to(self.device)
            hr_images = hr_images.to(self.device)
            # Enable Automatic mixed precision training
            # Automatic mixed precision training
            with torch.no_grad():
                output = self.model(lr_images)
                # To calculate the MSE loss
                # You need to use the standard normal distribution of x at time t and the predicted noise
                val_loss = self.loss_func(output, hr_images)
            # The optimizer clears the gradient of the model parameters
            self.optimizer.zero_grad()

            # TensorBoard logging
            val_pbar.set_postfix(MSE=val_loss.item())
            self.tb_logger.add_scalar(tag=f"[{self.device}]: Val loss({self.loss_func})", scalar_value=val_loss.item(),
                                      global_step=self.epoch * self.len_val_dataloader + i)
            val_loss_list.append(val_loss.item())

            # Metric
            ssim_res = compute_ssim(image_outputs=output, image_sources=hr_images)
            psnr_res = compute_psnr(mse=val_loss.item())
            self.tb_logger.add_scalar(tag=f"[{self.device}]: SSIM({self.loss_func})", scalar_value=ssim_res,
                                      global_step=self.epoch * self.len_val_dataloader + i)
            self.tb_logger.add_scalar(tag=f"[{self.device}]: PSNR({self.loss_func})", scalar_value=psnr_res,
                                      global_step=self.epoch * self.len_val_dataloader + i)
            ssim_list.append(ssim_res)
            psnr_list.append(psnr_res)

            # Save super resolution image and high resolution image
            lr_images = post_image(lr_images, device=self.device)
            sr_images = post_image(output, device=self.device)
            hr_images = post_image(hr_images, device=self.device)
            image_name = time.time()
            for lr_index, lr_image in enumerate(lr_images):
                save_images(images=lr_image,
                            path=os.path.join(self.save_val_vis_dir,
                                              f"{i}_{image_name}_{lr_index}_lr.{self.image_format}"))
            for sr_index, sr_image in enumerate(sr_images):
                save_images(images=sr_image,
                            path=os.path.join(self.save_val_vis_dir,
                                              f"{i}_{image_name}_{sr_index}_sr.{self.image_format}"))
            for hr_index, hr_image in enumerate(hr_images):
                save_images(images=hr_image,
                            path=os.path.join(self.save_val_vis_dir,
                                              f"{i}_{image_name}_{hr_index}_hr.{self.image_format}"))
        # Loss, ssim and psnr per epoch
        self.avg_val_loss = sum(val_loss_list) / len(val_loss_list)
        self.avg_ssim = sum(ssim_list) / len(ssim_list)
        self.avg_psnr = sum(psnr_list) / len(psnr_list)
        self.tb_logger.add_scalar(tag=f"[{self.device}]: Val loss", scalar_value=self.avg_val_loss,
                                  global_step=self.epoch)
        self.tb_logger.add_scalar(tag=f"[{self.device}]: Avg ssim", scalar_value=self.avg_ssim, global_step=self.epoch)
        self.tb_logger.add_scalar(tag=f"[{self.device}]: Avg psnr", scalar_value=self.avg_psnr, global_step=self.epoch)
        logger.info(f"Val loss: {self.avg_val_loss}, SSIM: {self.avg_ssim}, PSNR: {self.avg_psnr}")
        logger.info(msg="Finish val mode.")

    def after_iter(self):
        """
        After training one iter diffusion model method
        """
        # Saving and validating models in the main process
        if self.save_models:
            # Saving model, set the checkpoint name
            save_name = f"ckpt_{str(self.epoch).zfill(3)}"
            # Init ckpt params
            ckpt_model = self.model.state_dict()
            ckpt_ema_model = self.ema_model.state_dict()
            ckpt_optimizer = self.optimizer.state_dict()
            # Save the best model
            if (self.avg_ssim > self.best_ssim) and (self.avg_psnr > self.best_psnr):
                is_best = True
                self.best_ssim = self.avg_ssim
                self.best_psnr = self.avg_psnr
            else:
                is_best = False
            # Save checkpoint
            save_ckpt(epoch=self.epoch, save_name=save_name, ckpt_model=ckpt_model, ckpt_ema_model=ckpt_ema_model,
                      ckpt_optimizer=ckpt_optimizer, results_dir=self.results_dir,
                      save_model_interval=self.save_model_interval, start_model_interval=self.start_model_interval,
                      save_model_interval_epochs=self.save_model_interval_epochs, image_size=self.image_size,
                      network=self.network, act=self.act, is_sr=True, is_best=is_best, ssim=self.avg_ssim,
                      psnr=self.avg_psnr, best_ssim=self.best_ssim, best_psnr=self.best_psnr)

        super().after_iter()

    def after_train(self):
        """
        After training super resolution model method
        """
        super().after_train()
