#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Date   : 2024/1/24 17:11
    @Author : chairc
    @Site   : https://github.com/chairc
"""
import os
import sys
import argparse
import logging

import coloredlogs
import torch

from torch import multiprocessing as mp

sys.path.append(os.path.dirname(sys.path[0]))
from iddm.config.choices import sr_network_choices, optim_choices, sr_loss_func_choices, image_format_choices
from iddm.config.version import get_version_banner
from iddm.model.trainers.sr import SRTrainer

logger = logging.getLogger(__name__)
coloredlogs.install(level="INFO")


def main(args):
    """
    Main function
    :param args: Input parameters
    :return: None
    """
    if args.distributed:
        gpus = torch.cuda.device_count()
        mp.spawn(SRTrainer(args=args).train, nprocs=gpus)
    else:
        SRTrainer(args=args).train()


def init_sr_train_args():
    """
    Init super resolution model training arguments
    :return: args
    """
    # Training model parameters
    # required: Must be set
    # needed: Set as needed
    # recommend: Recommend to set
    parser = argparse.ArgumentParser()
    # =================================Base settings=================================
    # Set the seed for initialization (required)
    parser.add_argument("--seed", "-s", type=int, default=0)
    # Set network
    # Option: srv1
    parser.add_argument("--network", type=str, default="srv1", choices=sr_network_choices)
    # File name for initializing the model (required)
    parser.add_argument("--run_name", "-n", type=str, default="sr")
    # Total epoch for training (required)
    parser.add_argument("--epochs", "-e", type=int, default=300)
    # Batch size for training (required)
    parser.add_argument("--batch_size", "-b", type=int, default=8)
    # Number of sub-processes used for data loading (needed)
    # It may consume a significant amount of CPU and memory, but it can speed up the training process.
    parser.add_argument("--num_workers", type=int, default=2)
    # Input image size (required)
    parser.add_argument("--image_size", "-i", type=int, default=128)
    # Dataset path (required)
    # Conditional dataset
    # e.g: cifar10, Each category is stored in a separate folder, and the main folder represents the path.
    # Unconditional dataset
    # All images are placed in a single folder, and the path represents the image folder.
    parser.add_argument("--train_dataset_path", type=str,
                        default="/your/path/Diffusion-Model/datasets/dir/train")
    parser.add_argument("--val_dataset_path", type=str,
                        default="/your/path/Diffusion-Model/datasets/dir/val")
    # Enable automatic mixed precision training (needed)
    # Effectively reducing GPU memory usage may lead to lower training accuracy and results.
    parser.add_argument("--amp", default=False, action="store_true")
    # It is recommended that the batch size of the evaluation data be set to 1,
    # which can accurately evaluate each picture and reduce the evaluation error of each group of pictures.
    # If you want to evaluate quickly, set it to batch size.
    parser.add_argument("--quick_eval", default=False, action="store_true")
    # Set optimizer (needed)
    # Option: adam/adamw/sgd
    parser.add_argument("--optim", type=str, default="sgd", choices=optim_choices)
    # Set loss function
    # Option: mse only
    parser.add_argument("--loss", type=str, default="mse", choices=sr_loss_func_choices)
    # Set activation function (needed)
    # Option: gelu/silu/relu/relu6/lrelu
    parser.add_argument("--act", type=str, default="silu")
    # Learning rate (needed)
    parser.add_argument("--lr", type=float, default=1e-3)
    # Learning rate function (needed)
    # Option: linear/cosine/warmup_cosine
    parser.add_argument("--lr_func", type=str, default="cosine")
    # Saving path (required)
    parser.add_argument("--result_path", type=str, default="/your/path/Diffusion-Model/results")
    # Whether to save weight each training (recommend)
    parser.add_argument("--save_model_interval", default=False, action="store_true")
    # Save model interval and save it every X epochs (needed)
    parser.add_argument("--save_model_interval_epochs", type=int, default=10)
    # Start epoch for saving models (needed)
    # This option saves disk space. If not set, the default is '-1'. If set,
    # it starts saving models from the specified epoch. It needs to be used with '--save_model_interval'
    parser.add_argument("--start_model_interval", type=int, default=-1)
    # Generated image format
    # Recommend to use png for better generation quality.
    # Option: jpg/png
    parser.add_argument("--image_format", type=str, default="png", choices=image_format_choices)
    # Resume interrupted training (needed)
    # 1. Set to 'True' to resume interrupted training and check if the parameter 'run_name' is correct.
    # 2. Set the resume interrupted epoch number. (If not, we would select the last)
    # Note: If the epoch number of interruption is outside the condition of '--start_model_interval',
    # it will not take effect. For example, if the start saving model time is 100 and the interruption number is 50,
    # we cannot set any loading epoch points because we did not save the model.
    # We save the 'ckpt_last.pt' file every training, so we need to use the last saved model for interrupted training
    # If you do not know what epoch the checkpoint is, rename this checkpoint is 'ckpt_last'.pt
    parser.add_argument("--resume", "-r", default=False, action="store_true")
    parser.add_argument("--start_epoch", type=int, default=None)
    # Enable use pretrain model (needed)
    parser.add_argument("--pretrain", default=False, action="store_true")
    # Pretrain model load path (needed)
    parser.add_argument("--pretrain_path", type=str, default="")
    # Set the use GPU in normal training (required)
    parser.add_argument("--use_gpu", type=int, default=0)

    # =================================Enable distributed training (if applicable)=================================
    # Enable distributed training (needed)
    parser.add_argument("--distributed", "-d", default=False, action="store_true")
    # Set the main GPU (required)
    # Default GPU is '0'
    parser.add_argument("--main_gpu", type=int, default=0)
    # Number of distributed nodes (needed)
    # The value of world size will correspond to the actual number of GPUs or distributed nodes being used
    parser.add_argument("--world_size", type=int, default=2)

    return parser.parse_args()


if __name__ == "__main__":
    args = init_sr_train_args()
    # Get version banner
    get_version_banner()
    main(args)
