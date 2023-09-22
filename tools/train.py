#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Date   : 2023/6/20 22:33
    @Author : chairc
    @Site   : https://github.com/chairc
"""
import os
import sys
import argparse
import copy
import logging
import coloredlogs
import numpy as np
import torch

from torch import nn as nn
from torch import distributed as dist
from torch import multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

sys.path.append(os.path.dirname(sys.path[0]))
from model.ddpm import Diffusion as DDPMDiffusion
from model.ddim import Diffusion as DDIMDiffusion
from model.modules import EMA
from model.network import UNet
from utils.initializer import device_initializer, seed_initializer, load_model_weight_initializer
from utils.lr_scheduler import set_cosine_lr
from utils.utils import plot_images, save_images, get_dataset, setup_logging, save_train_logging

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
    # Sample type
    sample = args.sample
    # Run name
    run_name = args.run_name
    # Input image size
    image_size = args.image_size
    # Select optimizer
    optim = args.optim
    # Select activation function
    act = args.act
    # Learning rate
    init_lr = args.lr
    # Learning rate function
    lr_func = args.lr_func
    # Number of classes
    num_classes = args.num_classes
    # classifier-free guidance interpolation weight, users can better generate model effect
    cfg_scale = args.cfg_scale
    # Whether to enable conditional training
    conditional = args.conditional
    # Initialize and save the model identification bit
    # Check here whether it is single-GPU training or multi-GPU training
    save_models = True
    # Whether to enable distributed training
    if args.distributed and torch.cuda.device_count() > 1 and torch.cuda.is_available():
        distributed = True
        world_size = args.world_size
        # Set address and port
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12345"
        # The total number of processes is equal to the number of graphics cards
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo", rank=rank,
                                world_size=world_size)
        # Set device ID
        device = torch.device("cuda", rank)
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
        device = device_initializer()
        logger.info(msg=f"[{device}]: Successfully Use normal training.")
    # Whether to enable half-precision training
    fp16 = args.fp16
    # Save model interval
    save_model_interval = args.save_model_interval
    # Save model interval in the start epoch
    start_model_interval = args.start_model_interval
    # Enable data visualization
    vis = args.vis
    # Number of visualization images generated
    num_vis = args.num_vis
    # Saving path
    result_path = args.result_path
    # Create data logging path
    results_logging = setup_logging(save_path=result_path, run_name=run_name)
    results_dir = results_logging[1]
    results_vis_dir = results_logging[2]
    results_tb_dir = results_logging[3]
    # Dataloader
    dataloader = get_dataset(args=args, distributed=distributed)
    # Resume training
    resume = args.resume
    # Model
    if not conditional:
        model = UNet(device=device, image_size=image_size, act=act).to(device)
    else:
        model = UNet(num_classes=num_classes, device=device, image_size=image_size, act=act).to(device)
    # Distributed training
    if distributed:
        model = nn.parallel.DistributedDataParallel(module=model, device_ids=[device], find_unused_parameters=True)
    # Model optimizer
    if optim == "adam":
        optimizer = torch.optim.Adam(params=model.parameters(), lr=init_lr)
    else:
        optimizer = torch.optim.AdamW(params=model.parameters(), lr=init_lr)
    # Resume training
    if resume:
        load_model_dir = args.load_model_dir
        start_epoch = args.start_epoch
        # Load the previous model
        load_epoch = str(start_epoch - 1).zfill(3)
        model_path = os.path.join(result_path, load_model_dir, f"model_{load_epoch}.pt")
        optim_path = os.path.join(result_path, load_model_dir, f"optim_model_{load_epoch}.pt")
        load_model_weight_initializer(model=model, weight_path=model_path, device=device)
        logger.info(msg=f"[{device}]: Successfully load model model_{load_epoch}.pt")
        # Load the previous model optimizer
        optim_weights_dict = torch.load(f=optim_path, map_location=device)
        optimizer.load_state_dict(state_dict=optim_weights_dict)
        logger.info(msg=f"[{device}]: Successfully load optimizer optim_model_{load_epoch}.pt")
    else:
        start_epoch = 0
    if fp16:
        logger.info(msg=f"[{device}]: Fp16 training is opened.")
        # Used to scale gradients to prevent overflow
        scaler = GradScaler()
    else:
        logger.info(msg=f"[{device}]: Fp32 training.")
        scaler = None
    # Loss function
    mse = nn.MSELoss()
    # Initialize the diffusion model
    if sample == "ddpm":
        diffusion = DDPMDiffusion(img_size=image_size, device=device)
    elif sample == "ddim":
        diffusion = DDIMDiffusion(img_size=image_size, device=device)
    else:
        diffusion = DDPMDiffusion(img_size=image_size, device=device)
        logger.warning(msg=f"[{device}]: Setting sample error, we has been automatically set to ddpm.")
    # Tensorboard
    tb_logger = SummaryWriter(log_dir=results_tb_dir)
    # Train log
    save_train_logging(args, results_dir)
    # Number of dataset batches in the dataloader
    len_dataloader = len(dataloader)
    # Exponential Moving Average (EMA) may not be as dominant for single class as for multi class
    ema = EMA(beta=0.995)
    # EMA model
    ema_model = copy.deepcopy(model).eval().requires_grad_(False)

    logger.info(msg=f"[{device}]: Start training.")
    # Start iterating
    for epoch in range(start_epoch, args.epochs):
        logger.info(msg=f"[{device}]: Start epoch {epoch}:")
        # Set learning rate
        if lr_func == "cosine":
            current_lr = set_cosine_lr(optimizer=optimizer, current_epoch=epoch, max_epoch=args.epochs,
                                       lr_min=init_lr * 0.01, lr_max=init_lr, warmup=False)
        elif lr_func == "warmup_cosine":
            current_lr = set_cosine_lr(optimizer=optimizer, current_epoch=epoch, max_epoch=args.epochs,
                                       lr_min=init_lr * 0.01, lr_max=init_lr, warmup=True)
        else:
            current_lr = init_lr
        logger.info(msg=f"[{device}]: This epoch learning rate is {current_lr}")
        pbar = tqdm(dataloader)
        # Initialize images and labels
        images, labels = None, None
        for i, (images, labels) in enumerate(pbar):
            # The images are all resized in dataloader
            images = images.to(device)
            # Generates a tensor of size images.shape[0] * images.shape[0] randomly sampled time steps
            time = diffusion.sample_time_steps(images.shape[0]).to(device)
            # Add noise, return as x value at time t and standard normal distribution
            x_time, noise = diffusion.noise_images(x=images, time=time)
            # Enable half-precision training
            if fp16:
                # Half-precision training
                with autocast():
                    # Half-precision unconditional training
                    if not conditional:
                        # Half-precision unconditional model prediction
                        predicted_noise = model(x_time, time)
                    # Conditional training, need to add labels
                    else:
                        labels = labels.to(device)
                        # Random unlabeled hard training, using only time steps and no class information
                        if np.random.random() < 0.1:
                            labels = None
                        # Half-precision conditional model prediction
                        predicted_noise = model(x_time, time, labels)
                    # To calculate the MSE loss
                    # You need to use the standard normal distribution of x at time t and the predicted noise
                    loss = mse(noise, predicted_noise)
                # The optimizer clears the gradient of the model parameters
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            # Use full-precision
            else:
                # Full-precision unconditional training
                if not conditional:
                    # Full-precision unconditional model prediction
                    predicted_noise = model(x_time, time)
                # Conditional training, need to add labels
                else:
                    labels = labels.to(device)
                    # Random unlabeled hard training, using only time steps and no class information
                    if np.random.random() < 0.1:
                        labels = None
                    # Full-precision conditional model prediction
                    predicted_noise = model(x_time, time, labels)
                # To calculate the MSE loss
                # You need to use the standard normal distribution of x at time t and the predicted noise
                loss = mse(noise, predicted_noise)
                # The optimizer clears the gradient of the model parameters
                optimizer.zero_grad()
                # Automatically calculate gradients
                loss.backward()
                # The optimizer updates the parameters of the model
                optimizer.step()
            # EMA
            ema.step_ema(ema_model=ema_model, model=model)

            # TensorBoard logging
            pbar.set_postfix(MSE=loss.item())
            tb_logger.add_scalar(tag=f"[{device}]: MSE", scalar_value=loss.item(),
                                 global_step=epoch * len_dataloader + i)

        # Saving and validating models in the main process
        if save_models:
            # Saving model
            save_name = f"model_{str(epoch).zfill(3)}"
            if not conditional:
                # Saving pt files
                torch.save(obj=model.state_dict(), f=os.path.join(results_dir, f"model_last.pt"))
                torch.save(obj=optimizer.state_dict(), f=os.path.join(results_dir, f"optim_last.pt"))
                # Enable visualization
                if vis:
                    # images.shape[0] is the number of images in the current batch
                    n = num_vis if num_vis > 0 else images.shape[0]
                    sampled_images = diffusion.sample(model=model, n=n)
                    save_images(images=sampled_images, path=os.path.join(results_vis_dir, f"{save_name}.jpg"))
                # Saving pt files in epoch interval
                if save_model_interval and epoch > start_model_interval:
                    torch.save(obj=model.state_dict(), f=os.path.join(results_dir, f"{save_name}.pt"))
                    torch.save(obj=optimizer.state_dict(), f=os.path.join(results_dir, f"optim_{save_name}.pt"))
                    logger.info(msg=f"Save the {save_name}.pt, and optim_{save_name}.pt.")
                logger.info(msg="Save the model.")
            else:
                # Saving pt files
                torch.save(obj=model.state_dict(), f=os.path.join(results_dir, f"model_last.pt"))
                torch.save(obj=ema_model.state_dict(), f=os.path.join(results_dir, f"ema_model_last.pt"))
                torch.save(obj=optimizer.state_dict(), f=os.path.join(results_dir, f"optim_last.pt"))
                # Enable visualization
                if vis:
                    labels = torch.arange(num_classes).long().to(device)
                    n = num_vis if num_vis > 0 else len(labels)
                    sampled_images = diffusion.sample(model=model, n=n, labels=labels, cfg_scale=cfg_scale)
                    ema_sampled_images = diffusion.sample(model=ema_model, n=n, labels=labels,
                                                          cfg_scale=cfg_scale)
                    # This is a method to display the results of each model during training and can be commented out
                    # plot_images(images=sampled_images)
                    save_images(images=sampled_images, path=os.path.join(results_vis_dir, f"{save_name}.jpg"))
                    save_images(images=ema_sampled_images, path=os.path.join(results_vis_dir, f"{save_name}_ema.jpg"))
                if save_model_interval and epoch > start_model_interval:
                    torch.save(obj=model.state_dict(), f=os.path.join(results_dir, f"{save_name}.pt"))
                    torch.save(obj=ema_model.state_dict(), f=os.path.join(results_dir, f"ema_{save_name}.pt"))
                    torch.save(obj=optimizer.state_dict(), f=os.path.join(results_dir, f"optim_{save_name}.pt"))
                    logger.info(msg=f"Save the {save_name}.pt, ema_{save_name}.pt, and optim_{save_name}.pt.")
                logger.info(msg="Save the model.")
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
    # Set the seed for initialization (required)
    parser.add_argument("--seed", type=int, default=0)
    # Enable conditional training (required)
    # If enabled, you can modify the custom configuration.
    # For more details, please refer to the boundary line at the bottom.
    parser.add_argument("--conditional", type=bool, default=False)
    # Set the sample type (required)
    # If not set, the default is for 'ddpm'. You can set it to either 'ddpm' or 'ddim'.
    # Option: ddpm/ddim
    parser.add_argument("--sample", type=str, default="ddpm")
    # File name for initializing the model (required)
    parser.add_argument("--run_name", type=str, default="df")
    # Total epoch for training (required)
    parser.add_argument("--epochs", type=int, default=3)
    # Batch size for training (required)
    parser.add_argument("--batch_size", type=int, default=2)
    # Number of sub-processes used for data loading (needed)
    # It may consume a significant amount of CPU and memory, but it can speed up the training process.
    parser.add_argument("--num_workers", type=int, default=0)
    # Input image size (required)
    parser.add_argument("--image_size", type=int, default=64)
    # Dataset path (required)
    # Conditional dataset
    # e.g: cifar10, Each category is stored in a separate folder, and the main folder represents the path.
    # Unconditional dataset
    # All images are placed in a single folder, and the path represents the image folder.
    parser.add_argument("--dataset_path", type=str, default="/your/path/Defect-Diffusion-Model/datasets/dir")
    # Enable half-precision training (needed)
    # Effectively reducing GPU memory usage may lead to lower training accuracy and results.
    parser.add_argument("--fp16", type=bool, default=False)
    # Set optimizer (needed)
    # Option: adam/adamw
    parser.add_argument("--optim", type=str, default="adamw")
    # Set activation function (needed)
    # Option: gelu/silu/relu/relu6/lrelu
    parser.add_argument("--act", type=str, default="gelu")
    # Learning rate (needed)
    parser.add_argument("--lr", type=float, default=3e-4)
    # Learning rate function (needed)
    # Option: linear/cosine/warmup_cosine
    parser.add_argument("--lr_func", type=str, default="")
    # Saving path (required)
    parser.add_argument("--result_path", type=str, default="/your/path/Defect-Diffusion-Model/results")
    # Whether to save weight each training (recommend)
    parser.add_argument("--save_model_interval", type=bool, default=True)
    # Start epoch for saving models (needed)
    # This option saves disk space. If not set, the default is '-1'. If set,
    # it starts saving models from the specified epoch. It needs to be used with '--save_model_interval'
    parser.add_argument("--start_model_interval", type=int, default=-1)
    # Enable visualization of dataset information for model selection based on visualization (recommend)
    parser.add_argument("--vis", type=bool, default=True)
    # Number of visualization images generated (recommend)
    # If not filled, the default is the number of image classes (unconditional) or images.shape[0] (conditional)
    parser.add_argument("--num_vis", type=int, default=-1)
    # Resume interrupted training (needed)
    # 1. Set to 'True' to resume interrupted training.
    # 2. Set the resume interrupted epoch number
    # 3. Set the directory of the previous loaded model from the interrupted epoch.
    # Note: If the epoch number of interruption is outside the condition of '--start_model_interval',
    # it will not take effect. For example, if the start saving model time is 100 and the interruption number is 50,
    # we cannot set any loading epoch points because we did not save the model.
    # We save the 'xxx_last.pt' file every training, so we need to use the last saved model for interrupted training
    parser.add_argument("--resume", type=bool, default=False)
    parser.add_argument("--start_epoch", type=int, default=-1)
    parser.add_argument("--load_model_dir", type=str, default="")

    # =================================Enable distributed training (if applicable)=================================
    # Enable distributed training (needed)
    parser.add_argument("--distributed", type=bool, default=False)
    # Set the main GPU (required)
    # Default GPU is '0'
    parser.add_argument("--main_gpu", type=int, default=0)
    # Number of distributed nodes (needed)
    # The value of world size will correspond to the actual number of GPUs or distributed nodes being used
    parser.add_argument("--world_size", type=int, default=2)

    # =====================Enable the conditional generation (if '--conditional' is set to 'True')=====================
    # Number of classes (required)
    parser.add_argument("--num_classes", type=int, default=1)
    # classifier-free guidance interpolation weight, users can better generate model effect (recommend)
    parser.add_argument("--cfg_scale", type=int, default=3)

    args = parser.parse_args()

    main(args)
