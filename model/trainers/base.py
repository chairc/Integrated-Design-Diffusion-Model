#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Date   : 2024/9/18 9:59
    @Author : chairc
    @Site   : https://github.com/chairc
"""
import argparse
import logging
import coloredlogs

logger = logging.getLogger(__name__)
coloredlogs.install(level="INFO")


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
        self.before_iter()
        self.train_in_iter()
        self.after_iter()

    def before_iter(self):
        """
        Before training one iter method
        """
        pass

    def train_in_iter(self):
        """
        Train in one iter method
        """
        pass

    def after_iter(self):
        """
        After training one iter
        """
        pass

    def after_train(self):
        """
        After training method
        """
        pass
