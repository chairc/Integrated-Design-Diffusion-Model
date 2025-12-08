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
    @Date   : 2024/9/18 13:10
    @Author : chairc
    @Site   : https://github.com/chairc
"""
import os
import sys
import copy
import logging
import coloredlogs
import numpy as np
import torch

from torch import nn as nn
from torch.amp import autocast
from tqdm import tqdm

sys.path.append(os.path.dirname(sys.path[0]))
from iddm.config.setting import EMA_BETA, LATENT_CHANNEL, IMAGE_CHANNEL, IMAGE_SCALE
from iddm.model.modules.ema import EMA
from iddm.utils.check import check_image_size, check_pretrain_path
from iddm.utils.dataset import get_dataset
from iddm.utils.initializer import seed_initializer, network_initializer, optimizer_initializer, sample_initializer, \
    amp_initializer, loss_initializer, classes_initializer, autoencoder_network_initializer
from iddm.utils.utils import plot_images, save_images, download_model_pretrain_model
from iddm.utils.checkpoint import load_ckpt, save_ckpt
from iddm.model.trainers.base import Trainer

logger = logging.getLogger(__name__)
coloredlogs.install(level="INFO")


class DMTrainer(Trainer):
    """
    Diffusion model trainer
    """

    def __init__(self, **kwargs):
        """
        Initialize diffusion model trainer
        :param kwargs: Parameters of trainer
        """
        super(DMTrainer, self).__init__(**kwargs)
        # Can be set input params
        # Check args is None, and input kwargs in initialize diffusion model trainer
        # e.g. trainer = DMTrainer(run_name="dm-temp", dataset_path="/your/dataset/path/dir")
        # Run name
        self.run_name = self.check_args_and_kwargs(kwarg="run_name", default="df")
        # Whether to enable conditional training
        self.conditional = self.check_args_and_kwargs(kwarg="conditional", default=False)
        # Mode type
        self.mode = self.check_args_and_kwargs(kwarg="mode", default="class")
        # Sample type
        self.sample = self.check_args_and_kwargs(kwarg="sample", default="ddpm")
        # Dataset path
        self.dataset_path = self.check_args_and_kwargs(kwarg="dataset_path", default="")
        # Enable data visualization
        self.vis = self.check_args_and_kwargs(kwarg="vis", default=True)
        # Number of visualization images generated
        self.num_vis = self.check_args_and_kwargs(kwarg="num_vis", default=-1)
        # Noise schedule
        self.noise_schedule = self.check_args_and_kwargs(kwarg="noise_schedule", default="linear")
        # classifier-free guidance interpolation weight, users can better generate model effect
        self.cfg_scale = self.check_args_and_kwargs(kwarg="cfg_scale", default=3)

        # Parameters specific to Latent Diffusion
        self.latent = self.check_args_and_kwargs(kwarg="latent", default=False)
        self.autoencoder_image_size = None
        self.autoencoder_ckpt = self.check_args_and_kwargs(kwarg="autoencoder_ckpt", default="")
        self.autoencoder_network = self.check_args_and_kwargs(kwarg="autoencoder_network", default="autoencoder")
        self.latent_channels = LATENT_CHANNEL
        self.autoencoder = None

        # Default params
        self.in_channels, self.out_channels = IMAGE_CHANNEL, IMAGE_CHANNEL
        self.dataset_image_size = 64
        self.num_classes = None
        self.diffusion = None
        self.pbar = None
        self.dataloader = None
        self.len_dataloader = None

    def before_train(self):
        """
        Before training diffusion model method
        """
        # =================================Before training=================================
        logger.info(msg=f"[{self.rank}]: Start diffusion model training")
        # Output params to console
        logger.info(msg=f"[{self.rank}]: Input params: {self.args}")
        # Step1: Set path and create log
        # Create data logging path
        self.init_trainer_results_dir_and_log()

        # Step2: Get the parameters of the initializer and args
        # Initialize the seed
        seed_initializer(seed_id=self.seed)
        # Input image size
        self.image_size = check_image_size(image_size=self.image_size)
        self.dataset_image_size = self.image_size
        if self.latent:
            # Autoencoder image size
            if isinstance(self.image_size, (list, tuple)):
                self.autoencoder_image_size = [s * IMAGE_SCALE for s in self.image_size]
            else:
                self.autoencoder_image_size = self.image_size * IMAGE_SCALE
            self.in_channels, self.out_channels = LATENT_CHANNEL, LATENT_CHANNEL
            self.dataset_image_size = self.autoencoder_image_size
        # Number of classes
        self.num_classes = classes_initializer(dataset_path=self.dataset_path)
        # Initialize and save the model identification bit
        self.init_trainer_distributed()

        # =================================About model initializer=================================
        # Step3: Init model
        # Network
        dm_model = network_initializer(network=self.network, device=self.device)
        # Model
        if not self.conditional:
            self.model = dm_model(in_channel=self.in_channels, out_channel=self.out_channels, device=self.device,
                                  image_size=self.image_size, act=self.act).to(self.device)
        else:
            self.model = dm_model(mode=self.mode, in_channel=self.in_channels, out_channel=self.out_channels,
                                  num_classes=self.num_classes, device=self.device, image_size=self.image_size,
                                  act=self.act).to(self.device)
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
            self.start_epoch, _ = load_ckpt(ckpt_path=ckpt_path, model=self.model, device=self.device,
                                            optimizer=self.optimizer, is_train=True, is_distributed=self.distributed,
                                            is_resume=self.resume, conditional=self.conditional)
            logger.info(msg=f"[{self.device}]: Successfully load resume model checkpoint.")
        else:
            # Pretrain mode
            if self.pretrain:
                # TODO: If pretrain path is none, download the official pretrain model
                if check_pretrain_path(pretrain_path=self.pretrain_path):
                    # If you want to train on a specified data set, such as neu or cifar 10
                    # You can set the df_type to exp and add model_name="neu-cls" or model_name="cifar10"
                    self.pretrain_path = download_model_pretrain_model(pretrain_type="df", network=self.network,
                                                                       conditional=self.conditional,
                                                                       image_size=self.image_size, df_type="default")
                load_ckpt(ckpt_path=self.pretrain_path, model=self.model, device=self.device, is_train=True,
                          is_pretrain=self.pretrain, is_distributed=self.distributed, conditional=self.conditional)
                logger.info(msg=f"[{self.device}]: Successfully load pretrain model checkpoint.")
            self.start_epoch = 0
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
        self.dataloader = get_dataset(image_size=self.dataset_image_size, dataset_path=self.dataset_path,
                                      batch_size=self.batch_size, num_workers=self.num_workers,
                                      distributed=self.distributed)
        # Number of dataset batches in the dataloader
        self.len_dataloader = len(self.dataloader)

        # =================================About autoencoder and diffusion=================================
        # Step5: Set autoencoder
        # Loading autoencoder (fixed parameters, only used to encode images into latent space)
        if self.latent:
            ae_model = autoencoder_network_initializer(network=self.autoencoder_network, device=self.device)
            self.autoencoder = ae_model(latent_channels=self.latent_channels, device=self.device).to(self.device)
            load_ckpt(ckpt_path=self.autoencoder_ckpt, model=self.autoencoder, is_generate=True,
                      device=self.device)
            # Inference mode, no updating parameters
            self.autoencoder.eval()
        # Initialize the diffusion model
        self.diffusion = sample_initializer(sample=self.sample, image_size=self.image_size, device=self.device,
                                            schedule_name=self.noise_schedule, latent=self.latent,
                                            latent_channel=self.latent_channels, autoencoder=self.autoencoder)

    def before_iter(self):
        """
        Before training one iter diffusion model method
        """
        super().before_iter()
        # Initialize the tqdm progress bar
        self.pbar = tqdm(self.dataloader)

    def train_in_iter(self):
        """
        Train in one iter diffusion model method
        """
        # Initialize images and labels
        images, labels, loss_list = None, None, []
        for i, (images, labels) in enumerate(self.pbar):
            # The images are all resized in dataloader
            images = images.to(self.device)
            if self.latent:
                with torch.no_grad():
                    # Latent variable shape: [B, C, H/8, W/8]
                    z = self.autoencoder.encode(images)
                    mean, log_var = z
                    z = self.autoencoder.reparameterize(mean, log_var, sample=False)
                    # Scaling latent variables
                    images = z * self.autoencoder.scale_factor
            # Generates a tensor of size images.shape[0] randomly sampled time steps
            time = self.diffusion.sample_time_steps(images.shape[0]).to(self.device)
            # Add noise, return as x value at time t and standard normal distribution
            x_time, noise = self.diffusion.noise_images(x=images, time=time)
            # Enable Automatic mixed precision training
            # Automatic mixed precision training
            # Note: Pytorch version must > 1.10
            with autocast("cuda", enabled=self.amp):
                # Unconditional training
                if not self.conditional:
                    # Unconditional model prediction
                    predicted_noise = self.model(x_time, time)
                # Conditional training, need to add labels
                else:
                    labels = labels.to(self.device)
                    # Random unlabeled hard training, using only time steps and no class information
                    if np.random.random() < 0.1:
                        labels = None
                    # Conditional model prediction
                    predicted_noise = self.model(x_time, time, labels)
                # To calculate the MSE loss
                # You need to use the standard normal distribution of x at time t and the predicted noise
                loss = self.loss_func(noise, predicted_noise)
            # The optimizer clears the gradient of the model parameters
            self.optimizer.zero_grad()
            # Update loss and optimizer
            # Fp16 + Fp32
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            # EMA
            self.ema.step_ema(ema_model=self.ema_model, model=self.model)

            # TensorBoard logging
            self.pbar.set_postfix(MSE=loss.item())
            self.tb_logger.add_scalar(tag=f"[{self.device}]: MSE", scalar_value=loss.item(),
                                      global_step=self.epoch * self.len_dataloader + i)
            loss_list.append(loss.item())
        # Loss per epoch
        self.tb_logger.add_scalar(tag=f"[{self.device}]: Loss", scalar_value=sum(loss_list) / len(loss_list),
                                  global_step=self.epoch)

    def after_iter(self):
        """
        After training one iter diffusion model method
        """
        # Saving and validating models in the main process
        if self.save_models:
            # Saving model, set the checkpoint name
            save_name = f"ckpt_{str(self.epoch).zfill(3)}"
            # Init ckpt params
            mode, ckpt_model, ckpt_ema_model, ckpt_optimizer = None, None, None, None
            if not self.conditional:
                ckpt_model = self.model.state_dict()
                ckpt_optimizer = self.optimizer.state_dict()
                # Enable visualization
                if self.vis:
                    # images.shape[0] is the number of images in the current batch
                    n = self.num_vis if self.num_vis > 0 else self.batch_size
                    sampled_images = self.diffusion.sample(model=self.model, n=n)
                    save_images(images=sampled_images,
                                path=os.path.join(self.results_vis_dir, f"{save_name}.{self.image_format}"))
            else:
                mode = self.mode
                ckpt_model = self.model.state_dict()
                ckpt_ema_model = self.ema_model.state_dict()
                ckpt_optimizer = self.optimizer.state_dict()
                # Enable visualization
                if self.vis:
                    labels = torch.arange(self.num_classes).long().to(self.device)
                    n = self.num_vis if self.num_vis > 0 else len(labels)
                    sampled_images = self.diffusion.sample(model=self.model, n=n, labels=labels,
                                                           cfg_scale=self.cfg_scale)
                    ema_sampled_images = self.diffusion.sample(model=self.ema_model, n=n, labels=labels,
                                                               cfg_scale=self.cfg_scale)
                    # This is a method to display the results of each model during training and can be commented out
                    # plot_images(images=sampled_images)
                    save_images(images=sampled_images,
                                path=os.path.join(self.results_vis_dir, f"{save_name}.{self.image_format}"))
                    save_images(images=ema_sampled_images,
                                path=os.path.join(self.results_vis_dir, f"ema_{save_name}.{self.image_format}"))
            # Save checkpoint
            save_ckpt(epoch=self.epoch, save_name=save_name, ckpt_model=ckpt_model, ckpt_ema_model=ckpt_ema_model,
                      ckpt_optimizer=ckpt_optimizer, results_dir=self.results_dir,
                      save_model_interval=self.save_model_interval,
                      save_model_interval_epochs=self.save_model_interval_epochs,
                      start_model_interval=self.start_model_interval, conditional=self.conditional,
                      image_size=self.image_size, sample=self.sample, network=self.network, act=self.act,
                      num_classes=self.num_classes, latent=self.latent, mode=mode)

        super().after_iter()

    def after_train(self):
        """
        After training diffusion model method
        """
        super().after_train()
        logger.info(msg="[Note]: If you want to evaluate model quality, use 'FID_calculator.py' to evaluate.")
