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
import copy
import logging
import time

import coloredlogs
import torch

from torch import nn as nn
from torch import distributed as dist
from torch import multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast
from tqdm import tqdm

sys.path.append(os.path.dirname(sys.path[0]))
from config.choices import loss_func_choices, sr_network_choices, optim_choices
from config.setting import MASTER_ADDR, MASTER_PORT, EMA_BETA
from config.version import get_version_banner
from model.modules.ema import EMA
from utils.initializer import device_initializer, seed_initializer, sr_network_initializer, optimizer_initializer, \
    lr_initializer, amp_initializer, loss_initializer
from utils.utils import save_images, setup_logging, save_train_logging, check_and_create_dir
from utils.checkpoint import load_ckpt, save_ckpt
from sr.interface import post_image
from sr.dataset import get_sr_dataset

logger = logging.getLogger(__name__)
coloredlogs.install(level="INFO")


def train(rank=None, args=None):
    """
    Training
    :param rank: Device id
    :param args: Input parameters
    :return: None
    """
    logger.info(msg=f"[{rank}]: Input params: {args}")
    # Initialize the seed
    seed_initializer(seed_id=args.seed)
    # Network
    network = args.network
    # Run name
    run_name = args.run_name
    # Input image size
    image_size = args.image_size
    # Datasets
    train_dataset_path = args.train_dataset_path
    val_dataset_path = args.val_dataset_path
    # Batch size
    batch_size = args.batch_size
    # Number of workers
    num_workers = args.num_workers
    # Select optimizer
    optim = args.optim
    # Select activation function
    loss_func = args.loss
    # Select activation function
    act = args.act
    # Learning rate
    init_lr = args.lr
    # Learning rate function
    lr_func = args.lr_func
    # Initialize and save the model identification bit
    # Check here whether it is single-GPU training or multi-GPU training
    save_models = True
    # Whether to enable distributed training
    if args.distributed and torch.cuda.device_count() > 1 and torch.cuda.is_available():
        distributed = True
        world_size = args.world_size
        # Set address and port
        os.environ["MASTER_ADDR"] = MASTER_ADDR
        os.environ["MASTER_PORT"] = MASTER_PORT
        # The total number of processes is equal to the number of graphics cards
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo", rank=rank,
                                world_size=world_size)
        # Set device ID
        device = device_initializer(device_id=rank, is_train=True)
        # There may be random errors, using this function can reduce random errors in cudnn
        # torch.backends.cudnn.deterministic = True
        # Synchronization during distributed training
        dist.barrier()
        # If the distributed training is not the main GPU, the save model flag is False
        if dist.get_rank() != args.main_gpu:
            save_models = False
        logger.info(msg=f"[{device}]: Successfully Use distributed training.")
    else:
        distributed = False
        # Run device initializer
        device = device_initializer(device_id=args.use_gpu, is_train=True)
        logger.info(msg=f"[{device}]: Successfully Use normal training.")
    # Whether to enable automatic mixed precision training
    amp = args.amp
    # Save model interval
    save_model_interval = args.save_model_interval
    # Save model interval in the start epoch
    start_model_interval = args.start_model_interval
    # Saving path
    result_path = args.result_path
    # Create data logging path
    results_logging = setup_logging(save_path=result_path, run_name=run_name)
    results_dir = results_logging[1]
    results_vis_dir = results_logging[2]
    results_tb_dir = results_logging[3]
    # Dataloader
    train_dataloader = get_sr_dataset(image_size=image_size, dataset_path=train_dataset_path, batch_size=batch_size,
                                      num_workers=num_workers, distributed=distributed)
    val_dataloader = get_sr_dataset(image_size=image_size, dataset_path=val_dataset_path, batch_size=batch_size,
                                    num_workers=num_workers, distributed=distributed)
    # Resume training
    resume = args.resume
    # Pretrain
    pretrain = args.pretrain
    # Network
    Network = sr_network_initializer(network=network, device=device)
    # Model
    model = Network(act=act).to(device)
    # Distributed training
    if distributed:
        model = nn.parallel.DistributedDataParallel(module=model, device_ids=[device], find_unused_parameters=True)
    # Model optimizer
    optimizer = optimizer_initializer(model=model, optim=optim, init_lr=init_lr, device=device)
    # Resume training
    if resume:
        ckpt_path = None
        start_epoch = args.start_epoch
        # Determine which checkpoint to load
        # 'start_epoch' is correct
        if start_epoch is not None:
            ckpt_path = os.path.join(results_dir, f"ckpt_{str(start_epoch - 1).zfill(3)}.pt")
        # Parameter 'ckpt_path' is None in the train mode
        if ckpt_path is None:
            ckpt_path = os.path.join(results_dir, "ckpt_last.pt")
        start_epoch = load_ckpt(ckpt_path=ckpt_path, model=model, device=device, optimizer=optimizer,
                                is_distributed=distributed)
        logger.info(msg=f"[{device}]: Successfully load resume model checkpoint.")
    else:
        # Pretrain mode
        if pretrain:
            pretrain_path = args.pretrain_path
            load_ckpt(ckpt_path=pretrain_path, model=model, device=device, is_pretrain=pretrain,
                      is_distributed=distributed)
            logger.info(msg=f"[{device}]: Successfully load pretrain model checkpoint.")
        start_epoch = 0
    # Set harf-precision
    scaler = amp_initializer(amp=amp, device=device)
    # Loss function
    loss_func = loss_initializer(loss_name=loss_func, device=device)
    # Tensorboard
    tb_logger = SummaryWriter(log_dir=results_tb_dir)
    # Train log
    save_train_logging(args, results_dir)
    # Number of dataset batches in the dataloader
    len_train_dataloader = len(train_dataloader)
    len_val_dataloader = len(val_dataloader)
    # Exponential Moving Average (EMA) may not be as dominant for single class as for multi class
    ema = EMA(beta=EMA_BETA)
    # EMA model
    ema_model = copy.deepcopy(model).eval().requires_grad_(False)

    logger.info(msg=f"[{device}]: Start training.")
    # Start iterating
    for epoch in range(start_epoch, args.epochs):
        logger.info(msg=f"[{device}]: Start epoch {epoch}:")
        # Set learning rate
        current_lr = lr_initializer(lr_func=lr_func, optimizer=optimizer, epoch=epoch, epochs=args.epochs,
                                    init_lr=init_lr, device=device)
        tb_logger.add_scalar(tag=f"[{device}]: Current LR", scalar_value=current_lr, global_step=epoch)
        # Create vis dir
        save_val_vis_dir = os.path.join(results_vis_dir, str(epoch))
        check_and_create_dir(save_val_vis_dir)
        # Initialize images and labels
        train_loss_list, val_loss_list = [], []

        # Train
        model.train()
        logger.info(msg="Start train mode.")
        train_pbar = tqdm(train_dataloader)
        for i, (lr_images, hr_images) in enumerate(train_pbar):
            # The images are all resized in train dataloader
            lr_images = lr_images.to(device)
            hr_images = hr_images.to(device)
            # Enable Automatic mixed precision training
            # Automatic mixed precision training
            # Note: If your Pytorch version > 2.4.1, with torch.amp.autocast("cuda", enabled=amp):
            with autocast(enabled=amp):
                output = model(lr_images)
                # To calculate the MSE loss
                # You need to use the standard normal distribution of x at time t and the predicted noise
                train_loss = loss_func(output, hr_images)
            # The optimizer clears the gradient of the model parameters
            optimizer.zero_grad()
            # Update loss and optimizer
            # Fp16 + Fp32
            scaler.scale(train_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            # EMA
            ema.step_ema(ema_model=ema_model, model=model)

            # TensorBoard logging
            train_pbar.set_postfix(MSE=train_loss.item())
            tb_logger.add_scalar(tag=f"[{device}]: Train loss({loss_func})", scalar_value=train_loss.item(),
                                 global_step=epoch * len_train_dataloader + i)
            train_loss_list.append(train_loss.item())
        # Loss per epoch
        tb_logger.add_scalar(tag=f"[{device}]: Train loss", scalar_value=sum(train_loss_list) / len(train_loss_list),
                             global_step=epoch)
        logger.info(msg="Finish train mode.")

        # Val
        model.eval()
        logger.info(msg="Start val mode.")
        val_pbar = tqdm(val_dataloader)
        for i, (lr_images, hr_images) in enumerate(val_pbar):
            # The images are all resized in val dataloader
            lr_images = lr_images.to(device)
            hr_images = hr_images.to(device)
            # Enable Automatic mixed precision training
            # Automatic mixed precision training
            with torch.no_grad():
                output = model(lr_images)
                # To calculate the MSE loss
                # You need to use the standard normal distribution of x at time t and the predicted noise
                val_loss = loss_func(output, hr_images)
            # The optimizer clears the gradient of the model parameters
            optimizer.zero_grad()

            # TensorBoard logging
            val_pbar.set_postfix(MSE=val_loss.item())
            tb_logger.add_scalar(tag=f"[{device}]: Val loss({loss_func})", scalar_value=val_loss.item(),
                                 global_step=epoch * len_val_dataloader + i)
            val_loss_list.append(val_loss.item())
            # Save super resolution image and high resolution image
            lr_images = post_image(lr_images)
            sr_images = post_image(output)
            hr_images = post_image(hr_images)
            image_name = time.time()
            for lr_index, lr_image in enumerate(lr_images):
                save_images(images=lr_image, path=os.path.join(save_val_vis_dir, f"{i}_{image_name}_{lr_index}_lr.jpg"))
            for sr_index, sr_image in enumerate(sr_images):
                save_images(images=sr_image, path=os.path.join(save_val_vis_dir, f"{i}_{image_name}_{sr_index}_sr.jpg"))
            for hr_index, hr_image in enumerate(hr_images):
                save_images(images=hr_image, path=os.path.join(save_val_vis_dir, f"{i}_{image_name}_{hr_index}_hr.jpg"))
        # Loss per epoch
        tb_logger.add_scalar(tag=f"[{device}]: Val loss", scalar_value=sum(val_loss_list) / len(val_loss_list),
                             global_step=epoch)
        logger.info(msg="Finish val mode.")

        # Saving and validating models in the main process
        if save_models:
            # Saving model, set the checkpoint name
            save_name = f"ckpt_{str(epoch).zfill(3)}"
            # Init ckpt params
            ckpt_model = model.state_dict()
            ckpt_ema_model = ema_model.state_dict()
            ckpt_optimizer = optimizer.state_dict()
            # Save checkpoint
            save_ckpt(epoch=epoch, save_name=save_name, ckpt_model=ckpt_model, ckpt_ema_model=ckpt_ema_model,
                      ckpt_optimizer=ckpt_optimizer, results_dir=results_dir, save_model_interval=save_model_interval,
                      start_model_interval=start_model_interval, image_size=image_size, network=network, act=act)
        logger.info(msg=f"[{device}]: Finish epoch {epoch}:")

        # Synchronization during distributed training
        if distributed:
            logger.info(msg=f"[{device}]: Synchronization during distributed training.")
            dist.barrier()

    logger.info(msg=f"[{device}]: Finish training.")

    # Clean up the distributed environment
    if distributed:
        dist.destroy_process_group()


def main(args):
    """
    Main function
    :param args: Input parameters
    :return: None
    """
    if args.distributed:
        gpus = torch.cuda.device_count()
        mp.spawn(train, args=(args,), nprocs=gpus)
    else:
        train(args=args)


if __name__ == "__main__":
    # Training model parameters
    # required: Must be set
    # needed: Set as needed
    # recommend: Recommend to set
    parser = argparse.ArgumentParser()
    # =================================Base settings=================================
    # Set the seed for initialization (required)
    parser.add_argument("--seed", type=int, default=0)
    # Set network
    # Option: srv1
    parser.add_argument("--network", type=str, default="srv1", choices=sr_network_choices)
    # File name for initializing the model (required)
    parser.add_argument("--run_name", type=str, default="sr")
    # Total epoch for training (required)
    parser.add_argument("--epochs", type=int, default=300)
    # Batch size for training (required)
    parser.add_argument("--batch_size", type=int, default=8)
    # Number of sub-processes used for data loading (needed)
    # It may consume a significant amount of CPU and memory, but it can speed up the training process.
    parser.add_argument("--num_workers", type=int, default=2)
    # Input image size (required)
    parser.add_argument("--image_size", type=int, default=128)
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
    # Set optimizer (needed)
    # Option: adam/adamw/sgd
    parser.add_argument("--optim", type=str, default="sgd", choices=optim_choices)
    # Set loss function (needed)
    # Option: mse/l1/huber/smooth_l1
    parser.add_argument("--loss", type=str, default="mse", choices=loss_func_choices)
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
    # Start epoch for saving models (needed)
    # This option saves disk space. If not set, the default is '-1'. If set,
    # it starts saving models from the specified epoch. It needs to be used with '--save_model_interval'
    parser.add_argument("--start_model_interval", type=int, default=-1)
    # Resume interrupted training (needed)
    # 1. Set to 'True' to resume interrupted training and check if the parameter 'run_name' is correct.
    # 2. Set the resume interrupted epoch number. (If not, we would select the last)
    # Note: If the epoch number of interruption is outside the condition of '--start_model_interval',
    # it will not take effect. For example, if the start saving model time is 100 and the interruption number is 50,
    # we cannot set any loading epoch points because we did not save the model.
    # We save the 'ckpt_last.pt' file every training, so we need to use the last saved model for interrupted training
    # If you do not know what epoch the checkpoint is, rename this checkpoint is 'ckpt_last'.pt
    parser.add_argument("--resume", default=False, action="store_true")
    parser.add_argument("--start_epoch", type=int, default=None)
    # Enable use pretrain model (needed)
    parser.add_argument("--pretrain", default=False, action="store_true")
    # Pretrain model load path (needed)
    parser.add_argument("--pretrain_path", type=str, default="")
    # Set the use GPU in normal training (required)
    parser.add_argument("--use_gpu", type=int, default=0)

    # =================================Enable distributed training (if applicable)=================================
    # Enable distributed training (needed)
    parser.add_argument("--distributed", default=False, action="store_true")
    # Set the main GPU (required)
    # Default GPU is '0'
    parser.add_argument("--main_gpu", type=int, default=0)
    # Number of distributed nodes (needed)
    # The value of world size will correspond to the actual number of GPUs or distributed nodes being used
    parser.add_argument("--world_size", type=int, default=2)

    args = parser.parse_args()
    # Get version banner
    get_version_banner()
    main(args)
