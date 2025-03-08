#!/usr/bin/env python
# -*- coding:utf-8 -*-
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
from torch import distributed as dist
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast
from tqdm import tqdm

sys.path.append(os.path.dirname(sys.path[0]))
from iddm.config.setting import MASTER_ADDR, MASTER_PORT, EMA_BETA
from iddm.model.modules.ema import EMA
from iddm.utils.check import check_image_size, check_pretrain_path, check_is_distributed
from iddm.utils.dataset import get_dataset
from iddm.utils.initializer import device_initializer, seed_initializer, network_initializer, optimizer_initializer, \
    sample_initializer, lr_initializer, amp_initializer, loss_initializer, classes_initializer
from iddm.utils.utils import plot_images, save_images, setup_logging, save_train_logging, download_model_pretrain_model
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

        # Default params
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
        # Network
        Network = network_initializer(network=self.network, device=self.device)
        # Model
        if not self.conditional:
            self.model = Network(device=self.device, image_size=self.image_size, act=self.act).to(self.device)
        else:
            self.model = Network(num_classes=self.num_classes, device=self.device, image_size=self.image_size,
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
            self.start_epoch = load_ckpt(ckpt_path=ckpt_path, model=self.model, device=self.device,
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
        # Initialize the diffusion model
        self.diffusion = sample_initializer(sample=self.sample, image_size=self.image_size, device=self.device,
                                            schedule_name=self.noise_schedule)
        # Exponential Moving Average (EMA) may not be as dominant for single class as for multi class
        self.ema = EMA(beta=EMA_BETA)
        # EMA model
        self.ema_model = copy.deepcopy(self.model).eval().requires_grad_(False)

        # =================================About data=================================
        # Step4: Set data
        # Dataloader
        self.dataloader = get_dataset(image_size=self.image_size, dataset_path=self.dataset_path,
                                      batch_size=self.batch_size, num_workers=self.num_workers,
                                      distributed=self.distributed)
        # Number of dataset batches in the dataloader
        self.len_dataloader = len(self.dataloader)

    def train_in_epochs(self):
        """
        Train in epochs diffusion model method
        """
        # Step5: Training
        logger.info(msg=f"[{self.device}]: Start training.")
        # Start iterating
        for self.epoch in range(self.start_epoch, self.epochs):
            self.before_iter()
            self.train_in_iter()
            self.after_iter()

    def before_iter(self):
        """
        Before training one iter diffusion model method
        """
        logger.info(msg=f"[{self.device}]: Start epoch {self.epoch}:")
        # Set learning rate
        current_lr = lr_initializer(lr_func=self.lr_func, optimizer=self.optimizer, epoch=self.epoch,
                                    epochs=self.epochs, init_lr=self.init_lr, device=self.device)
        self.tb_logger.add_scalar(tag=f"[{self.device}]: Current LR", scalar_value=current_lr, global_step=self.epoch)
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
            # Generates a tensor of size images.shape[0] randomly sampled time steps
            time = self.diffusion.sample_time_steps(images.shape[0]).to(self.device)
            # Add noise, return as x value at time t and standard normal distribution
            x_time, noise = self.diffusion.noise_images(x=images, time=time)
            # Enable Automatic mixed precision training
            # Automatic mixed precision training
            # Note: If your Pytorch version > 2.4.1, with torch.amp.autocast("cuda", enabled=amp):
            with autocast(enabled=self.amp):
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

    def after_train(self):
        """
        After training diffusion model method
        """
        logger.info(msg=f"[{self.device}]: Finish training.")
        logger.info(msg="[Note]: If you want to evaluate model quality, use 'FID_calculator.py' to evaluate.")

        # Clean up the distributed environment
        if self.distributed:
            dist.destroy_process_group()
