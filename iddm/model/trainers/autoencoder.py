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
    @Date   : 2025/07/28 15:15
    @Author : chairc
    @Site   : https://github.com/chairc
"""
import os
import sys
import logging
import time

import coloredlogs
import torch

from torch import nn as nn
from torch.amp import autocast
from tqdm import tqdm

sys.path.append(os.path.dirname(sys.path[0]))
from iddm.config.setting import LATENT_CHANNEL
from iddm.model.trainers.base import Trainer
from iddm.utils.checkpoint import load_ckpt, save_ckpt
from iddm.utils.initializer import seed_initializer, optimizer_initializer, amp_initializer, loss_initializer, \
    autoencoder_network_initializer
from iddm.utils.utils import check_and_create_dir, save_images
from iddm.utils.dataset import get_dataset, post_image

logger = logging.getLogger(__name__)
coloredlogs.install(level="INFO")


class AutoencoderTrainer(Trainer):
    """
    Autoencoder Trainer
    """

    def __init__(self, **kwargs):
        super(AutoencoderTrainer, self).__init__(**kwargs)
        # Autoencoder parameters
        self.run_name = self.check_args_and_kwargs(kwarg="run_name", default="autoencoder")
        # Datasets
        self.train_dataset_path = self.check_args_and_kwargs(kwarg="train_dataset_path", default="")
        self.val_dataset_path = self.check_args_and_kwargs(kwarg="val_dataset_path", default="")
        # Latent space parameters
        self.latent_channels = LATENT_CHANNEL

        # Default params
        self.train_dataloader = None
        self.val_dataloader = None
        self.len_train_dataloader = None
        self.len_val_dataloader = None
        self.save_val_vis_dir = None
        self.best_score = 0
        self.avg_train_loss = 0
        self.avg_val_loss = 0
        self.avg_score = 0

    def before_train(self):
        """
        Before training autoencoder model method
        """
        # =================================Before training=================================
        logger.info(msg=f"[{self.rank}]: Start autoencoder model training")
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
        ae_model = autoencoder_network_initializer(network=self.network, device=self.device)
        self.model = ae_model(latent_channels=self.latent_channels, device=self.device).to(self.device)
        # Distributed training
        if self.distributed:
            self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[self.device],
                                                             find_unused_parameters=True)
        # Model optimizer
        self.optimizer = optimizer_initializer(model=self.model, optim=self.optim, init_lr=self.init_lr,
                                               device=self.device)
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
                                                      ckpt_type="autoencoder")
            self.best_score = model_score[0]
            logger.info(msg=f"[{self.device}]: Successfully load resume model checkpoint.")
        else:
            # Pretrain mode
            if self.pretrain:
                load_ckpt(ckpt_path=self.pretrain_path, model=self.model, device=self.device, is_train=True,
                          is_pretrain=self.pretrain, is_distributed=self.distributed, ckpt_type="autoencoder")
                logger.info(msg=f"[{self.device}]: Successfully load pretrain model checkpoint.")
            self.start_epoch, self.best_score = 0, 0
        logger.info(msg=f"[{self.device}]: The start epoch is {self.start_epoch}, best score is {self.best_score}.")

        # Set harf-precision
        self.scaler = amp_initializer(amp=self.amp, device=self.device)
        # Loss function
        self.loss_func = loss_initializer(loss_name=self.loss_name, device=self.device)

        # =================================About data=================================
        # Step4: Set data
        self.train_dataloader = get_dataset(image_size=self.image_size, dataset_path=self.train_dataset_path,
                                            batch_size=self.batch_size, num_workers=self.num_workers,
                                            distributed=self.distributed)
        self.val_dataloader = get_dataset(image_size=self.image_size, dataset_path=self.val_dataset_path,
                                          batch_size=self.batch_size, num_workers=self.num_workers,
                                          distributed=self.distributed)
        # Number of dataset batches in the dataloader
        self.len_train_dataloader = len(self.train_dataloader)
        self.len_val_dataloader = len(self.val_dataloader)

    def before_iter(self):
        """
        Before training one iter autoencoder model method
        """
        super().before_iter()
        # Create vis dir
        self.save_val_vis_dir = os.path.join(self.results_vis_dir, str(self.epoch))
        check_and_create_dir(self.save_val_vis_dir)

    def train_in_iter(self):
        """
        Train in one iter autoencoder model method
        """
        # Initialize images and labels
        train_loss_list, val_loss_list, score_list = [], [], []
        # Train
        self.model.train()
        logger.info(msg="Start train mode.")
        train_pbar = tqdm(self.train_dataloader)
        for i, (images, _) in enumerate(train_pbar):
            # Input images [B, C, H, W]
            images = images.to(self.device)
            # Enable Automatic mixed precision training
            # Automatic mixed precision training
            # Note: Pytorch version must > 1.10
            with torch.amp.autocast("cuda", enabled=self.amp):
                recon_images = self.model(images)
                # To calculate the MSE loss
                train_loss = self.loss_func(recon_images, images)

            # The optimizer clears the gradient of the model parameters
            self.optimizer.zero_grad()
            # Update loss and optimizer
            # Fp16 + Fp32
            self.scaler.scale(train_loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # TensorBoard logging
            train_pbar.set_postfix(MSE=train_loss.item())
            self.tb_logger.add_scalar(tag=f"[{self.device}]: Train loss({self.loss_func})",
                                      scalar_value=train_loss.item(),
                                      global_step=self.epoch * self.len_train_dataloader + i)
            train_loss_list.append(train_loss.item())
            # Loss per epoch
        self.avg_train_loss = sum(train_loss_list) / len(train_loss_list)
        self.tb_logger.add_scalar(tag=f"[{self.device}]: Train loss",
                                  scalar_value=self.avg_train_loss,
                                  global_step=self.epoch)
        logger.info(f"Train loss: {self.avg_train_loss}")
        logger.info(msg="Finish train mode.")

        # Val
        self.model.eval()
        logger.info(msg="Start val mode.")
        val_pbar = tqdm(self.val_dataloader)
        for i, (images, _) in enumerate(val_pbar):
            # Input images [B, C, H, W]
            images = images.to(self.device)

            with autocast(device_type="cuda", enabled=self.amp):
                recon_images = self.model(images)
                # To calculate the MSE loss
                val_loss = self.loss_func(recon_images, images)

            # The optimizer clears the gradient of the model parameters
            self.optimizer.zero_grad()

            # TensorBoard logging
            val_pbar.set_postfix(MSE=val_loss.item())
            self.tb_logger.add_scalar(tag=f"[{self.device}]: Val loss({self.loss_func})", scalar_value=val_loss.item(),
                                      global_step=self.epoch * self.len_val_dataloader + i)
            val_loss_list.append(val_loss.item())

            # Metric
            score = 0
            self.tb_logger.add_scalar(tag=f"[{self.device}]: Score({self.loss_func})", scalar_value=score,
                                      global_step=self.epoch * self.len_val_dataloader + i)
            score_list.append(score)

            images = post_image(images=images, device=self.device)
            if self.loss_name == "mse_kl":
                recon_images = recon_images[0]
            recon_images = post_image(images=recon_images, device=self.device)
            image_name = time.time()
            for index, image in enumerate(images):
                save_images(images=image,
                            path=os.path.join(self.save_val_vis_dir,
                                              f"{i}_{image_name}_{index}_origin.{self.image_format}"))
            for recon_index, recon_image in enumerate(recon_images):
                save_images(images=recon_image,
                            path=os.path.join(self.save_val_vis_dir,
                                              f"{i}_{image_name}_{recon_index}_recon.{self.image_format}"))
        # Loss, score per epoch
        self.avg_val_loss = sum(val_loss_list) / len(val_loss_list)
        self.avg_score = sum(score_list) / len(score_list)
        self.tb_logger.add_scalar(tag=f"[{self.device}]: Val loss", scalar_value=self.avg_val_loss,
                                  global_step=self.epoch)
        self.tb_logger.add_scalar(tag=f"[{self.device}]: Avg score", scalar_value=self.avg_score,
                                  global_step=self.epoch)
        logger.info(f"Val loss: {self.avg_val_loss}, Score: {self.avg_score}")
        logger.info(msg="Finish val mode.")

    def after_iter(self):
        """
        After training one iter autoencoder model method
        """
        # Saving and validating models in the main process
        if self.save_models:
            # Saving model, set the checkpoint name
            save_name = f"ckpt_{str(self.epoch).zfill(3)}"
            # Init ckpt params
            ckpt_model = self.model.state_dict()
            ckpt_optimizer = self.optimizer.state_dict()
            # Save the best model
            if self.avg_score > self.best_score:
                is_best = True
                self.best_score = self.avg_score
            else:
                is_best = False
            save_ckpt(epoch=self.epoch, save_name=save_name, ckpt_model=ckpt_model, ckpt_ema_model=None,
                      ckpt_optimizer=ckpt_optimizer, results_dir=self.results_dir,
                      save_model_interval=self.save_model_interval,
                      save_model_interval_epochs=self.save_model_interval_epochs,
                      start_model_interval=self.start_model_interval, image_size=self.image_size,
                      network=self.network, act=self.act, is_autoencoder=True, is_best=is_best, score=self.avg_score,
                      best_score=self.best_score, latent_channel=self.latent_channels)
        super().after_iter()

    def after_train(self):
        """
        After training autoencoder model method
        """
        super().after_train()
