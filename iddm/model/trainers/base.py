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
    @Date   : 2024/9/18 9:59
    @Author : chairc
    @Site   : https://github.com/chairc
"""
import os
import sys
import torch
import argparse

from torch import distributed as dist
from torch.utils.tensorboard import SummaryWriter

sys.path.append(os.path.dirname(sys.path[0]))
from iddm.config.setting import MASTER_ADDR, MASTER_PORT
from iddm.utils.check import check_is_distributed
from iddm.utils.initializer import device_initializer, lr_initializer

from iddm.utils.utils import setup_logging, save_train_logging
from iddm.utils.logger import get_logger

logger = get_logger(name=__name__)


class Trainer:
    """
    Base trainer
    """

    def __init__(self, args=None, **kwargs):
        """
        Initialize trainer
        :param args: Args parser
        :param kwargs: Parameters of trainer
        """
        # Can be set input params
        self.args, self.kwargs, self.args_flag, self.rank = args, kwargs, False, None
        # Check input params valid
        if not kwargs and args is None:
            raise ValueError("Trainer must provide arguments")
        # New argparse
        if self.args is None:
            self.args_flag = True
            self.args = argparse.ArgumentParser().parse_args()
            logger.info(msg="[Note]: Trainer initializer successfully. But 'args' is None")

        # Random seed
        self.seed = self.check_args_and_kwargs(kwarg="seed", default=0)
        # Run name
        self.run_name = self.check_args_and_kwargs(kwarg="run_name", default="iddm")
        # Network
        self.network = self.check_args_and_kwargs(kwarg="network", default="unet")
        # Batch size
        self.batch_size = self.check_args_and_kwargs(kwarg="batch_size", default=2)
        # Number of workers
        self.num_workers = self.check_args_and_kwargs(kwarg="num_workers", default=0)
        # Input image size
        self.image_size = self.check_args_and_kwargs(kwarg="image_size", default=64)
        # Number of epochs
        self.epochs = self.check_args_and_kwargs(kwarg="epochs", default=300)
        # Whether to enable automatic mixed precision training
        self.amp = self.check_args_and_kwargs(kwarg="amp", default=False)
        # Select optimizer
        self.optim = self.check_args_and_kwargs(kwarg="optim", default="adamw")
        # Loss function
        self.loss_name = self.check_args_and_kwargs(kwarg="loss", default="mse")
        # Select activation function
        self.act = self.check_args_and_kwargs(kwarg="act", default="gelu")
        # Learning rate
        self.init_lr = self.check_args_and_kwargs(kwarg="lr", default=3e-4)
        # Learning rate function
        self.lr_func = self.check_args_and_kwargs(kwarg="lr_func", default="linear")
        # Saving path
        self.result_path = self.check_args_and_kwargs(kwarg="result_path", default="")
        # Save model interval
        self.save_model_interval = self.check_args_and_kwargs(kwarg="save_model_interval", default=False)
        # Save model interval and save it every X epochs
        self.save_model_interval_epochs = self.check_args_and_kwargs(kwarg="save_model_interval_epochs", default=10)
        # Save model interval in the start epoch
        self.start_model_interval = self.check_args_and_kwargs(kwarg="start_model_interval", default=-1)
        # Save image format
        self.image_format = self.check_args_and_kwargs(kwarg="image_format", default="png")
        # Resume training
        self.resume = self.check_args_and_kwargs(kwarg="resume", default=False)
        # Resume training epoch num
        self.start_epoch = self.check_args_and_kwargs(kwarg="start_epoch", default=-1)
        # Pretrain
        self.pretrain = self.check_args_and_kwargs(kwarg="pretrain", default=False)
        # Pretrain path
        self.pretrain_path = self.check_args_and_kwargs(kwarg="pretrain_path", default="")
        # Set the use GPU in normal training
        self.use_gpu = self.check_args_and_kwargs(kwarg="use_gpu", default=0)
        # Enable distributed training
        self.distributed = self.check_args_and_kwargs(kwarg="distributed", default=False)
        # Set the main GPU
        self.main_gpu = self.check_args_and_kwargs(kwarg="main_gpu", default=0)
        # Number of distributed node
        self.world_size = self.check_args_and_kwargs(kwarg="world_size", default=2)

        # Default params
        self.results_dir = None
        self.results_tb_dir = None
        self.results_logging = None
        self.results_vis_dir = None
        self.device = None
        self.save_models = None
        self.model = None
        self.ema = None
        self.ema_model = None
        self.epoch = None
        self.optimizer = None
        self.scaler = None
        self.loss_func = None
        self.tb_logger = None

    def check_args_and_kwargs(self, kwarg, default=None):
        """
        Check args with **kwargs
        :param kwarg: **kwargs params
        :param default: Default params
        :return: Used params
        """
        # Prevent loading parameters from failing and call default values
        if self.args_flag:
            value = self.kwargs.get(kwarg, default)
        else:
            # Get the self.args
            arg = getattr(self.args, kwarg)
            value = self.kwargs.get(kwarg, arg)
        # Load the params
        if self.kwargs.get(kwarg) is not None or self.args_flag:
            # The value of kwargs modifies the value of args
            setattr(self.args, kwarg, value)
            logger.info(msg=f"[Note]: args.{kwarg} already set => {value}")
        return value

    def train(self, rank=None):
        """
        Training method
        :param rank: Device id
        """
        # Init rank
        self.rank = rank

        # Training
        self.before_train()
        self.train_in_epochs()
        self.after_train()

    def before_train(self):
        """
        Before training method
        """
        pass

    def train_in_epochs(self):
        """
        Train in epochs method
        """
        logger.info(msg=f"[{self.device}]: Start training.")
        # Start iterating
        for self.epoch in range(self.start_epoch, self.epochs):
            self.before_iter()
            self.train_in_iter()
            self.after_iter()

    def before_iter(self):
        """
        Before training one iter method
        """
        logger.info(msg=f"[{self.device}]: Start epoch {self.epoch}:")
        # Set learning rate
        current_lr = lr_initializer(lr_func=self.lr_func, optimizer=self.optimizer, epoch=self.epoch,
                                    epochs=self.epochs, init_lr=self.init_lr, device=self.device)
        self.tb_logger.add_scalar(tag=f"[{self.device}]: Current LR", scalar_value=current_lr, global_step=self.epoch)

    def train_in_iter(self):
        """
        Train in one iter method
        """
        pass

    def after_iter(self):
        """
        After training one iter
        """
        logger.info(msg=f"[{self.device}]: Finish epoch {self.epoch}:")

        # Synchronization during distributed training
        self.synchronized_trainer_distributed()

    def after_train(self):
        """
        After training method
        """
        logger.info(msg=f"[{self.device}]: Finish training.")

        # Clean up the distributed environment
        self.destroy_trainer_distributed()

    def init_trainer_results_dir_and_log(self):
        """
        Initialize results directory
        """
        self.results_logging = setup_logging(save_path=self.result_path, run_name=self.run_name)
        self.results_dir = self.results_logging[1]
        self.results_vis_dir = self.results_logging[2]
        self.results_tb_dir = self.results_logging[3]

        # Tensorboard
        self.tb_logger = SummaryWriter(log_dir=self.results_tb_dir)
        # Train log
        self.args = save_train_logging(arg=self.args, save_path=self.results_dir)

    def init_trainer_distributed(self):
        """
        Initialize distributed training
        """
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

    def synchronized_trainer_distributed(self):
        """
        Synchronized distributed training
        """
        if self.distributed:
            logger.info(msg=f"[{self.device}]: Synchronization during distributed training.")
            dist.barrier()

    def destroy_trainer_distributed(self):
        """
        Destroy distributed training
        """
        if self.distributed:
            dist.destroy_process_group()
            logger.info(msg=f"[{self.device}]: Successfully destroy distributed training.")
