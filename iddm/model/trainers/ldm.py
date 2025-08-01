#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Date   : 2025/07/28 14:50
    @Author : chairc
    @Site   : https://github.com/chairc
"""
import os
import sys
import copy
import logging
import coloredlogs

import torch
from torch import nn as nn
from torch import distributed as dist
from torch.cuda.amp import autocast
from torch.utils.tensorboard import SummaryWriter

from iddm.model.samples.ldm import LDMDiffusion

sys.path.append(os.path.dirname(sys.path[0]))
from iddm.config.setting import MASTER_ADDR, MASTER_PORT, EMA_BETA
from iddm.model.modules.ema import EMA
from iddm.utils.check import check_image_size, check_pretrain_path, check_is_distributed
from iddm.utils.dataset import get_dataset
from iddm.config.setting import LATENT_CHANNEL, IMAGE_SCALE
from iddm.model.trainers.dm import DMTrainer
from iddm.utils.checkpoint import load_ckpt, save_ckpt
from iddm.utils.initializer import device_initializer, seed_initializer, network_initializer, optimizer_initializer, \
    sample_initializer, autoencoder_network_initializer, amp_initializer, loss_initializer, classes_initializer
from iddm.utils.utils import setup_logging, save_train_logging, download_model_pretrain_model, save_images

logger = logging.getLogger(__name__)
coloredlogs.install(level="INFO")


class LDMTrainer(DMTrainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Parameters specific to Latent Diffusion
        self.autoencoder_image_size = None
        self.autoencoder_ckpt = self.check_args_and_kwargs(kwarg="autoencoder_ckpt", default="")
        self.autoencoder_network = self.check_args_and_kwargs(kwarg="autoencoder_network", default="autoencoder")
        self.latent_channels = LATENT_CHANNEL
        self.autoencoder = None

    def before_train(self):
        """
        Initialize before training
        """
        # =================================Before training=================================
        logger.info(msg=f"[{self.rank}]: Start diffusion model training")
        # Output params to console
        logger.info(msg=f"[{self.rank}]: Input params: {self.args}")
        # Step1: Set path and create log
        # Create data logging path
        self.results_logging = setup_logging(save_path=self.result_path, run_name=self.run_name)
        self.results_dir = self.results_logging[1]
        self.results_vis_dir = self.results_logging[2]
        self.results_tb_dir = self.results_logging[3]
        # Tensorboard
        self.tb_logger = SummaryWriter(log_dir=self.results_tb_dir)
        # Train log
        self.args = save_train_logging(arg=self.args, save_path=self.results_dir)

        # Step2: Get the parameters of the initializer and args
        # Initialize the seed
        seed_initializer(seed_id=self.seed)
        # Input image size
        self.image_size = check_image_size(image_size=self.image_size)
        # Autoencoder image size
        if isinstance(self.image_size, (list, tuple)):
            self.autoencoder_image_size = [s * IMAGE_SCALE for s in self.image_size]
        else:
            self.autoencoder_image_size = self.image_size * IMAGE_SCALE
        # Number of classes
        self.num_classes = classes_initializer(dataset_path=self.dataset_path)
        # Initialize and save the model identification bit
        # Check here whether it is single-GPU training or multi-GPU training
        self.save_models = True
        # Whether to enable distributed training
        if check_is_distributed(distributed=self.distributed):
            self.distributed = True
            # Set address and port
            os.environ["MASTER_ADDR"] = MASTER_ADDR
            os.environ["MASTER_PORT"] = MASTER_PORT
            # The total number of processes is equal to the number of graphics cards
            dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo", rank=self.rank,
                                    world_size=self.world_size)
            # Set device ID
            self.device = device_initializer(device_id=self.rank, is_train=True)
            # There may be random errors, using this function can reduce random errors in cudnn
            # torch.backends.cudnn.deterministic = True
            # Synchronization during distributed training
            dist.barrier()
            # If the distributed training is not the main GPU, the save model flag is False
            if dist.get_rank() != self.main_gpu:
                self.save_models = False
            logger.info(msg=f"[{self.device}]: Successfully Use distributed training.")
        else:
            self.distributed = False
            # Run device initializer
            self.device = device_initializer(device_id=self.use_gpu, is_train=True)
            logger.info(msg=f"[{self.device}]: Successfully Use normal training.")

        # =================================About model initializer=================================
        # Step3: Init model
        # Diffusion network
        Network = network_initializer(network=self.network, device=self.device)
        # Diffusion unet model
        if not self.conditional:
            self.model = Network(in_channel=LATENT_CHANNEL, out_channel=LATENT_CHANNEL, device=self.device,
                                 image_size=self.image_size, act=self.act).to(self.device)
        else:
            self.model = Network(in_channel=LATENT_CHANNEL, out_channel=LATENT_CHANNEL, num_classes=self.num_classes,
                                 device=self.device, image_size=self.image_size,
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
                                            optimizer=self.optimizer, is_distributed=self.distributed,
                                            conditional=self.conditional)
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
                load_ckpt(ckpt_path=self.pretrain_path, model=self.model, device=self.device, is_pretrain=self.pretrain,
                          is_distributed=self.distributed, conditional=self.conditional)
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
        # Reset the input channel and output channel to the latent channel
        self.dataloader = get_dataset(image_size=self.autoencoder_image_size, dataset_path=self.dataset_path,
                                      batch_size=self.batch_size, num_workers=self.num_workers,
                                      distributed=self.distributed)
        # Number of dataset batches in the dataloader
        self.len_dataloader = len(self.dataloader)

        # =================================About autoencoder=================================
        # Step4: Set autoencoder
        # Loading autoencoder (fixed parameters, only used to encode images into subspatial space)
        Network = autoencoder_network_initializer(network=self.autoencoder_network, device=self.device)
        self.autoencoder = Network(latent_channels=self.latent_channels, device=self.device).to(self.device)
        load_ckpt(ckpt_path=self.autoencoder_ckpt, model=self.autoencoder, is_train=False, device=self.device)
        # Inference mode, no updating parameters
        self.autoencoder.eval()
        # Initialize the diffusion model
        self.diffusion = LDMDiffusion(autoencoder=self.autoencoder, img_size=self.image_size,
                                      device=self.device, schedule_name=self.noise_schedule)

    def train_in_iter(self):
        """
        Rewrite the iterative training logic and use latent variables as diffusion model inputs
        """
        self.model.train()
        for i, (images, labels) in enumerate(self.pbar):
            images = images.to(self.device)
            # 1. Use an autoencoder to encode the image as a latent variable
            with torch.no_grad():
                # Latent variable shape: [B, C, H/8, W/8]
                z = self.autoencoder.encode(images)
                mean, log_var = z
                z = self.autoencoder.reparameterize(mean, log_var, sample=False)
                # Scaling latent variables
                z_scaled = z * self.autoencoder.scale_factor

            # 2. Diffusion model training (similar to DMTrainer, but with an input of z)
            sample_time = self.diffusion.sample_time_steps(self.batch_size).to(self.device)
            z_noisy, noise = self.diffusion.noise_images(x=z_scaled, time=sample_time)

            with autocast(enabled=self.amp):
                if not self.conditional:
                    predicted_noise = self.model(z_noisy, sample_time)
                else:
                    labels = labels.to(self.device)
                    predicted_noise = self.model(z_noisy, sample_time, labels)
                loss = self.loss_func(noise, predicted_noise)

            # 3. Backpropagation
            # EMA
            self.ema.step_ema(ema_model=self.ema_model, model=self.model)
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Log updates
            self.pbar.set_postfix(LDMLoss=loss.item())
            self.tb_logger.add_scalar("LDMLoss", loss.item(), global_step=self.epoch * self.len_dataloader + i)

    def after_iter(self):
        """
        After training one iter latent diffusion model method
        """
        # Saving and validating models in the main process
        if self.save_models:
            # Saving model, set the checkpoint name
            save_name = f"ckpt_{str(self.epoch).zfill(3)}"
            # Init ckpt params
            ckpt_model, ckpt_ema_model, ckpt_optimizer = None, None, None
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
                      num_classes=self.num_classes)
        logger.info(msg=f"[{self.device}]: Finish epoch {self.epoch}:")

        # Synchronization during distributed training
        if self.distributed:
            logger.info(msg=f"[{self.device}]: Synchronization during distributed training.")
            dist.barrier()
