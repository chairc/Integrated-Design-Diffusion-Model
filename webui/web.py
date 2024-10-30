#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Date   : 2024/1/21 15:13
    @Author : chairc
    @Site   : https://github.com/chairc
"""
import os
import sys
import copy
import time

import gradio
import webbrowser
import numpy as np
import torch
import torchvision
from PIL import Image

from torch import nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast
from tqdm import tqdm

sys.path.append(os.path.dirname(sys.path[0]))
from config.choices import bool_choices, sample_choices, network_choices, optim_choices, act_choices, lr_func_choices, \
    image_format_choices
from config.setting import RANDOM_RESIZED_CROP_SCALE, MEAN, STD
from model.modules.ema import EMA
from utils.initializer import device_initializer, seed_initializer, network_initializer, optimizer_initializer, \
    sample_initializer, lr_initializer, amp_initializer, classes_initializer
from utils.dataset import get_dataset
from utils.utils import plot_images, save_images, setup_logging, check_and_create_dir
from utils.checkpoint import load_ckpt, save_ckpt
from utils.logger import CustomLogger


class GradioWebui:
    def __init__(self):
        # Kill train() flag
        self.KILL_FLAG = True

    def train(self, seed, conditional, sample, network, run_name, epochs, batch_size, num_workers, image_size,
              dataset_path, amp, optim, act, lr, lr_func, result_path, save_model_interval,
              start_model_interval, vis, num_vis, resume, start_epoch, pretrain, pretrain_path, use_gpu,
              distributed, main_gpu, world_size, image_format, cfg_scale):
        """
        0: seed
        1: conditional
        2: sample
        3: network
        4: run_name
        5: epochs
        6: batch_size
        7: num_workers
        8: image_size
        9: dataset_path
        10: amp
        11: optim
        12: act
        13: lr
        14: lr_func
        15: result_path
        16: save_model_interval
        17: start_model_interval
        18: vis
        19: num_vis
        20: resume
        21: start_epoch
        22: pretrain
        23: pretrain_path
        24: use_gpu
        25: distributed
        26: main_gpu
        27: world_size
        28: image_format
        29: cfg_scale
        """
        gradio.Info(message="Start training...")
        logger = CustomLogger(name=__name__, is_webui=True, is_save_log=True,
                              log_path=os.path.join(result_path, run_name))
        self.KILL_FLAG = False
        yield logger.info(msg="[Note]: Start parameters setting.")
        yield logger.info(
            msg=f"[Note]: params: {seed}, {conditional}, {sample}, {network}, {run_name}, {epochs}, {batch_size}, "
                f"{num_workers}, {image_size}, {dataset_path}, {amp}, {optim}, {act}, {lr}, {lr_func}, "
                f"{result_path}, {save_model_interval}, {start_model_interval}, {vis}, {num_vis}, {resume}, "
                f"{start_epoch}, {pretrain}, {pretrain_path}, {use_gpu}, {distributed}, {main_gpu}, "
                f"{world_size}, {image_format}, {cfg_scale}")

        # yield logger.info(msg=f"[Note]: Input params: {args}")
        # Initialize the seed
        seed_initializer(seed_id=seed)
        # Learning rate
        init_lr = lr
        # Run device initializer
        device = device_initializer(device_id=int(use_gpu), is_train=True)
        num_classes = classes_initializer(dataset_path=dataset_path)
        yield logger.info(msg=f"[{device}]: Successfully Use normal training.")
        # Create data logging path
        results_logging = setup_logging(save_path=result_path, run_name=run_name)
        results_dir = results_logging[1]
        results_vis_dir = results_logging[2]
        results_tb_dir = results_logging[3]
        # Load the folder data under the current path,
        # and automatically divide the labels according to the dataset under each file name
        dataloader = get_dataset(image_size=image_size, dataset_path=dataset_path, batch_size=batch_size,
                                 num_workers=num_workers, distributed=distributed)
        # Network
        Network = network_initializer(network=network, device=device)
        # Model
        if not conditional:
            model = Network(device=device, image_size=image_size, act=act).to(device)
        else:
            model = Network(num_classes=num_classes, device=device, image_size=image_size, act=act).to(device)
        # Model optimizer
        optimizer = optimizer_initializer(model=model, optim=optim, init_lr=init_lr, device=device)
        # Resume training
        if resume:
            ckpt_path = None
            # Determine which checkpoint to load
            # 'start_epoch' is correct
            if start_epoch > 0:
                ckpt_path = os.path.join(results_dir, f"ckpt_{str(start_epoch - 1).zfill(3)}.pt")
            # Parameter 'ckpt_path' is None in the train mode
            if ckpt_path is None:
                ckpt_path = os.path.join(results_dir, "ckpt_last.pt")
            start_epoch = load_ckpt(ckpt_path=ckpt_path, model=model, device=device, optimizer=optimizer,
                                    is_distributed=False)
            yield logger.info(msg=f"[{device}]: Successfully load resume model checkpoint.")
        else:
            # Pretrain mode
            if pretrain:
                load_ckpt(ckpt_path=pretrain_path, model=model, device=device, is_pretrain=pretrain,
                          is_distributed=False)
                yield logger.info(msg=f"[{device}]: Successfully load pretrain model checkpoint.")
            start_epoch = 0
        # Set harf-precision
        scaler = amp_initializer(amp=amp, device=device)
        # Loss function
        mse = nn.MSELoss()
        # Initialize the diffusion model
        diffusion = sample_initializer(sample=sample, image_size=image_size, device=device)
        # Tensorboard
        tb_logger = SummaryWriter(log_dir=results_tb_dir)
        # Number of dataset batches in the dataloader
        len_dataloader = len(dataloader)
        # Exponential Moving Average (EMA) may not be as dominant for single class as for multi class
        ema = EMA(beta=0.995)
        # EMA model
        ema_model = copy.deepcopy(model).eval().requires_grad_(False)

        yield logger.info(msg=f"[Note]: Start training.")
        # Start iterating
        for epoch in range(start_epoch, epochs):
            yield logger.info(msg=f"[{device}]: Start epoch {epoch}.")
            # Set learning rate
            current_lr = lr_initializer(lr_func=lr_func, optimizer=optimizer, epoch=epoch, epochs=epochs,
                                        init_lr=init_lr, device=device)
            tb_logger.add_scalar(tag=f"[{device}]: Current LR", scalar_value=current_lr, global_step=epoch)
            yield logger.info(msg=f"[{device}]: Current learning rate is {current_lr}.")
            pbar = tqdm(dataloader)
            # Initialize images and labels
            images, labels, loss_list = None, None, []
            for i, (images, labels) in enumerate(pbar):
                if self.KILL_FLAG:
                    yield logger.warning(msg=f"[Note]: Interrupt training.")
                    return
                # The images are all resized in dataloader
                images = images.to(device)
                # Generates a tensor of size images.shape[0] randomly sampled time steps
                time = diffusion.sample_time_steps(images.shape[0]).to(device)
                # Add noise, return as x value at time t and standard normal distribution
                x_time, noise = diffusion.noise_images(x=images, time=time)
                # Enable Automatic mixed precision training
                # Automatic mixed precision training
                # Note: If your Pytorch version > 2.4.1, with torch.amp.autocast("cuda", enabled=amp):
                with autocast(enabled=amp):
                    # Unconditional training
                    if not conditional:
                        # Unconditional model prediction
                        predicted_noise = model(x_time, time)
                    # Conditional training, need to add labels
                    else:
                        labels = labels.to(device)
                        # Random unlabeled hard training, using only time steps and no class information
                        if np.random.random() < 0.1:
                            labels = None
                        # Conditional model prediction
                        predicted_noise = model(x_time, time, labels)
                    # To calculate the MSE loss
                    # You need to use the standard normal distribution of x at time t and the predicted noise
                    loss = mse(noise, predicted_noise)
                # The optimizer clears the gradient of the model parameters
                optimizer.zero_grad()
                # Update loss and optimizer
                # Fp16 + Fp32
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                # EMA
                ema.step_ema(ema_model=ema_model, model=model)

                # TensorBoard logging
                pbar.set_postfix(MSE=loss.item())
                tb_logger.add_scalar(tag=f"[{device}]: MSE", scalar_value=loss.item(),
                                     global_step=epoch * len_dataloader + i)
                loss_list.append(loss.item())
            # Loss per epoch
            tb_logger.add_scalar(tag=f"[{device}]: Loss", scalar_value=sum(loss_list) / len(loss_list),
                                 global_step=epoch)
            yield logger.info(msg=f"[{device}]: Mean loss is {sum(loss_list) / len(loss_list)}.")

            # Saving and validating models in the main process
            # Saving model, set the checkpoint name
            save_name = f"ckpt_{str(epoch).zfill(3)}"
            # Init ckpt params
            ckpt_model, ckpt_ema_model, ckpt_optimizer = None, None, None
            if not conditional:
                ckpt_model = model.state_dict()
                ckpt_optimizer = optimizer.state_dict()
                # Enable visualization
                if vis:
                    yield logger.info(msg=f"[{device}]: Sampling...")
                    # images.shape[0] is the number of images in the current batch
                    n = num_vis if num_vis > 0 else images.shape[0]
                    sampled_images = diffusion.sample(model=model, n=n)
                    save_images(images=sampled_images,
                                path=os.path.join(results_vis_dir, f"{save_name}.{image_format}"))
                    yield logger.info(msg=f"[{device}]: Finish sample and generate images.")
            else:
                ckpt_model = model.state_dict()
                ckpt_ema_model = ema_model.state_dict()
                ckpt_optimizer = optimizer.state_dict()
                # Enable visualization
                if vis:
                    yield logger.info(msg=f"[{device}]: Sampling...")
                    labels = torch.arange(num_classes).long().to(device)
                    n = num_vis if num_vis > 0 else len(labels)
                    sampled_images = diffusion.sample(model=model, n=n, labels=labels, cfg_scale=cfg_scale)
                    ema_sampled_images = diffusion.sample(model=ema_model, n=n, labels=labels, cfg_scale=cfg_scale)
                    # This is a method to display the results of each model during training and can be commented out
                    # plot_images(images=sampled_images)
                    save_images(images=sampled_images,
                                path=os.path.join(results_vis_dir, f"{save_name}.{image_format}"))
                    save_images(images=ema_sampled_images,
                                path=os.path.join(results_vis_dir, f"ema_{save_name}.{image_format}"))
                    yield logger.info(msg=f"[{device}]: Finish sample and generate images.")

            # Save checkpoint
            save_ckpt(epoch=epoch, save_name=save_name, ckpt_model=ckpt_model, ckpt_ema_model=ckpt_ema_model,
                      ckpt_optimizer=ckpt_optimizer, results_dir=results_dir, save_model_interval=save_model_interval,
                      start_model_interval=start_model_interval, conditional=conditional, image_size=image_size,
                      sample=sample, network=network, act=act, num_classes=num_classes)
            yield logger.info(msg=f"[{device}]: Save epoch {epoch} checkpoint.")
            yield logger.info(msg=f"[{device}]: Finish epoch {epoch}.")

        yield logger.info(msg=f"[Note]: Finish training.")

    def cancer_train(self):
        gradio.Warning(message="Interrupt training.")
        self.KILL_FLAG = True

    def generate(self, generate_name, sample, image_format, use_ema, num_images, class_name, cfg_scale,
                 weight_path, result_path):

        """
        0: generate_name
        1: sample
        2: image_format
        3: use_ema
        4: num_images
        5: class_name
        6: cfg_scale
        7: weight_path
        8: result_path
        """
        gradio.Info(message="Start generation.")
        # Saving path
        result_path = os.path.join(result_path, str(time.time()))
        logger = CustomLogger(name=__name__, is_webui=True, is_save_log=True, log_path=os.path.join(result_path))
        # Check model params
        logger.info(msg="Start generation.")
        logger.info(
            msg=f"[Note]: params: {generate_name}, {sample}, {image_format}, {use_ema}, {num_images}, "
                f"{class_name}, {cfg_scale}, {weight_path}, {result_path}")
        # Run device initializer
        device = device_initializer()
        ckpt_state = torch.load(f=weight_path, map_location=device)
        conditional = ckpt_state["conditional"]
        network = ckpt_state["network"]
        image_size = ckpt_state["image_size"]
        act = ckpt_state["act"]
        num_classes = ckpt_state["num_classes"]
        # Check and create result path
        check_and_create_dir(result_path)
        # Network
        Network = network_initializer(network=network, device=device)
        # Initialize the diffusion model
        diffusion = sample_initializer(sample=sample, image_size=image_size, device=device)
        # Initialize model
        if conditional:
            model = Network(num_classes=num_classes, device=device, image_size=image_size, act=act).to(device)
            load_ckpt(ckpt_path=weight_path, model=model, device=device, is_train=False, is_use_ema=use_ema,
                      conditional=conditional)
            if class_name == -1:
                y = torch.arange(num_classes).long().to(device)
                num_images = num_classes
            else:
                y = torch.Tensor([class_name] * num_images).long().to(device)
            x = diffusion.sample(model=model, n=num_images, labels=y, cfg_scale=cfg_scale)
        else:
            model = Network(device=device, image_size=image_size, act=act).to(device)
            load_ckpt(ckpt_path=weight_path, model=model, device=device, is_train=False, conditional=conditional)
            x = diffusion.sample(model=model, n=num_images)
        # If there is no path information, it will only be displayed
        # If it exists, it will be saved to the specified path and displayed
        if result_path == "" or result_path is None:
            # plot_images(images=x)
            pass
        else:
            save_images(images=x, path=os.path.join(result_path, f"{generate_name}.{image_format}"))
            # plot_images(images=x)
        logger.info(msg="Finish generation.")
        gradio.Info(message="Finish generation.")
        return Image.open(fp=os.path.join(result_path, f"{generate_name}.{image_format}"))


def main():
    # Gradio layer
    webui = gradio.Blocks()
    webui.title = "IDDM-webui"
    # Init GradioWebui
    gradio_webui = GradioWebui()
    with webui:
        # Top
        with gradio.Row():
            gradio.HTML("<center><h1><b>IDDM-webui</b></h1></center>")
        with gradio.Row():
            gradio.HTML("<p style='font-size: 15px; text-align: left;'>"
                        "Author: <a href='https://github.com/chairc', chairc"
                        "target='_black'>chairc</a></p>")
        with gradio.Row():
            gradio.HTML(
                "<p style='font-size: 15px; text-align: left;'>"
                "Readme: <a href='https://github.com/chairc/Integrated-Design-Diffusion-Model/blob/main/README.md', "
                "target='_black'>https://github.com/chairc/Integrated-Design-Diffusion-Model</a></p>")
        with gradio.Row():
            gradio.HTML(
                "<p style='font-size: 15px; text-align: left;'>"
                "Note: The current version is a <b>webui beta version</b> and only supports 1 GPU training "
                "and some method implementations.</p>")

        # Mid
        with gradio.Tab(label="Train mode"):
            with gradio.Row():
                # Mid left
                with gradio.Column():
                    with gradio.Row():
                        gradio.HTML(
                            "<h3><b>Enable the normal training</b></h3>")
                    with gradio.Row():
                        # Set the seed for initialization
                        seed = gradio.Slider(minimum=0, maximum=10000, value=0, label="Seed",
                                             info="Set the seed for initialization (required).")
                        # File name for initializing the model
                        run_name = gradio.Textbox(label="Run name",
                                                  info="File name for initializing the model (required).",
                                                  lines=1, value="df")
                    with gradio.Row():
                        # Enable conditional training
                        conditional = gradio.Radio(choices=bool_choices, value=True, label="Conditional training",
                                                   info="Enable conditional training (required)."
                                                        "If enabled, you can modify the custom configuration."
                                                        "For more details, please refer to the boundary line at the bottom."
                                                        "[Note] We recommend enabling it to 'True'.")

                        # Number of classes
                        num_classes = gradio.Slider(minimum=1, maximum=1000, value=10, label="Number of classes",
                                                    info="Number of classes (required)."
                                                         "[Note] If enable the conditional training, "
                                                         "the classes settings must be consistent with the loaded "
                                                         "datasets settings.")
                    with gradio.Row():
                        # Set the sample type
                        sample = gradio.Radio(choices=sample_choices, value=sample_choices[0], label="Sample",
                                              info="Set the sample type (required)."
                                                   "If not set, the default is for 'ddpm'. "
                                                   f"Option: {sample_choices}")
                        # Input image size
                        image_size = gradio.Slider(minimum=32, maximum=512, step=2, value=64, label="Image size",
                                                   info="Set the seed for initialization (required).")

                    with gradio.Row():
                        # Image formate
                        image_format = gradio.Dropdown(choices=image_format_choices, value=image_format_choices[0],
                                                       label="Learning rate function",
                                                       info="Learning rate function (needed)."
                                                            " Option: linear/cosine/warmup_cosine")
                    with gradio.Row():
                        # Set network
                        network = gradio.Dropdown(choices=network_choices, value=network_choices[0], label="Network",
                                                  info=f"Set network (required). Option: {network_choices}")
                        # Set activation function
                        act = gradio.Dropdown(choices=act_choices, value=act_choices[0], label="Activation function",
                                              info=f"Set activation function (needed). Option: {act_choices}")
                    # Total epoch for training
                    epochs = gradio.Slider(minimum=1, maximum=1000, value=300, label="Epochs",
                                           info="Total epoch for training (required).")
                    with gradio.Row():
                        # Batch size for training
                        batch_size = gradio.Slider(minimum=2, maximum=256, step=2, value=2, label="Batch size",
                                                   info="Batch size for training (required).")
                        # Number of sub-processes used for data loading
                        num_workers = gradio.Slider(minimum=2, maximum=256, step=2, value=0, label="Num workers",
                                                    info="Number of sub-processes used for data loading (needed)."
                                                         "It may consume a significant amount of CPU and memory,"
                                                         "but it can speed up the training process.")
                    # Dataset path
                    dataset_path = gradio.Textbox(label="Dataset path", lines=1,
                                                  placeholder="/your/path/Defect-Diffusion-Model/datasets/dir",
                                                  info="[Conditional dataset] e.g: cifar10, Each category is stored in "
                                                       "a separate folder, and the main folder represents the path."
                                                       "[Unconditional dataset] All images are placed in a single folder, "
                                                       "and the path represents the image folder.")
                    # Saving path
                    result_path = gradio.Textbox(label="Result path", lines=1,
                                                 placeholder="/your/path/Defect-Diffusion-Model/results",
                                                 info="Saving path (required).")

                    with gradio.Row():
                        # Enable automatic mixed precision training
                        amp = gradio.Radio(choices=bool_choices, value=False,
                                           label="Automatic mixed precision training",
                                           info="Enable automatic mixed precision training (needed)."
                                                "Effectively reducing GPU memory usage may lead to "
                                                "lower training accuracy and results.")
                        # Set optimizer
                        optim = gradio.Dropdown(choices=optim_choices, value=optim_choices[0], label="Optimizer",
                                                info=f"Set optimizer (needed). Option: {optim_choices}")
                    with gradio.Row():
                        # Learning rate
                        lr = gradio.Slider(minimum=0, maximum=1e-1, step=1e-4, value=3e-4, label="Learning rate",
                                           info="Learning rate (needed).")
                        # Learning rate function
                        lr_func = gradio.Dropdown(choices=lr_func_choices, value=lr_func_choices[0],
                                                  label="Learning rate function",
                                                  info="Learning rate function (needed). "
                                                       f"Option: {lr_func_choices}")
                    with gradio.Row():
                        # Whether to save weight each training
                        save_model_interval = gradio.Radio(choices=bool_choices, value=True,
                                                           label="Save weight each training",
                                                           info="Whether to save weight each training (recommend).")
                        # Start epoch for saving models
                        start_model_interval = gradio.Slider(minimum=-1, maximum=1000, value=-1,
                                                             label="Start epoch for saving models",
                                                             info="Start epoch for saving models (needed)."
                                                                  "This option saves disk space. "
                                                                  "If not set, the default is '-1'. If set, "
                                                                  "it starts saving models from the specified epoch. "
                                                                  "It needs to be used with '--save_model_interval'")
                    with gradio.Row():
                        # Enable visualization
                        vis = gradio.Radio(choices=bool_choices, value=True, label="Visualization",
                                           info="Enable visualization of dataset information for model "
                                                "selection based on visualization (recommend).")
                        # Number of visualization images generated
                        num_vis = gradio.Slider(minimum=-1, maximum=100, step=1, value=-1,
                                                label="Number of visualization images",
                                                info="Number of visualization images generated (recommend)."
                                                     "If not filled, the default is the number of image "
                                                     "classes (unconditional) or images.shape[0] (conditional).")
                    with gradio.Row():
                        # Resume interrupted training
                        resume = gradio.Radio(choices=bool_choices, value=False, label="Resume training",
                                              info="Resume interrupted training (needed).")
                        # Restart epoch for training
                        start_epoch = gradio.Slider(minimum=-1, maximum=1000, value=-1, label="Epochs",
                                                    info="Resume epoch for training (needed).")
                    with gradio.Row():
                        # Enable use pretrain model
                        pretrain = gradio.Radio(choices=bool_choices, value=False, label="Pretrain model",
                                                info="Enable use pretrain model (needed).")
                        # Pretrain model load path
                        pretrain_path = gradio.Textbox(label="Pretrain path", lines=1,
                                                       placeholder="/your/path/Defect-Diffusion-Model/results/dir",
                                                       info="Pretrain model load path (needed).")
                    with gradio.Row():
                        # Set the use GPU in normal training
                        use_gpu = gradio.Textbox(label="Use GPU id", lines=1, value="0",
                                                 info="Set the use GPU in normal training (required).")
                        # classifier-free guidance
                        cfg_scale = gradio.Slider(minimum=1, maximum=100, value=3, label="Classifier-free guidance",
                                                  info="classifier-free guidance interpolation weight, "
                                                       "users can better generate model effect (recommend).")
                    with gradio.Row():
                        gradio.HTML("<h3><b>Enable distributed training (if applicable)</b></h3>", visible=False)
                    with gradio.Row():
                        # Enable distributed training
                        distributed = gradio.Radio(choices=bool_choices, value=False, label="Distributed training",
                                                   info="Enable distributed training (needed).", visible=False)
                        # Set the main GPU
                        main_gpu = gradio.Textbox(label="Use main GPU id", lines=1, value="0", visible=False,
                                                  info="Set the main GPU (required). Default GPU is '0'.")
                        # world size
                        world_size = gradio.Textbox(label="Number of distributed nodes", lines=1, value="2",
                                                    visible=False,
                                                    info="Number of distributed nodes (needed)."
                                                         "The value of world size will correspond to the actual "
                                                         "number of GPUs or distributed nodes being used.")

                # Mid right
                with gradio.Column():
                    with gradio.Row():
                        gradio.HTML(
                            "<h3><b>Results</b></h3>")
                    text_output = gradio.TextArea(label="Results of output", lines=23, autoscroll=True)
                    btn_start = gradio.Button("Start training")
                    btn_stop = gradio.Button("Stop training")
                    args = [seed, conditional, sample, network, run_name, epochs, batch_size, num_workers, image_size,
                            dataset_path, amp, optim, act, lr, lr_func, result_path, save_model_interval,
                            start_model_interval, vis, num_vis, resume, start_epoch, pretrain, pretrain_path, use_gpu,
                            distributed, main_gpu, world_size, image_format, cfg_scale, ]

                    # Train
                    btn_start.click(fn=gradio_webui.train, inputs=args, outputs=text_output)
                    btn_stop.click(fn=gradio_webui.cancer_train)
        with gradio.Tab(label="Generate mode"):
            with gradio.Row():
                with gradio.Column():
                    with gradio.Row():
                        generate_name = gradio.Textbox(label="Generate name", info="Generate name (required).",
                                                       lines=1, value="df-generate")
                        sample = gradio.Radio(choices=sample_choices, value=sample_choices[0], label="Sample",
                                              info="Set the sample type (required)."
                                                   "If not set, the default is for 'ddpm'. "
                                                   f"Option: {sample_choices}")
                    with gradio.Row():
                        image_format = gradio.Dropdown(choices=image_format_choices, value=image_format_choices[0],
                                                       label="Learning rate function",
                                                       info="Learning rate function (needed)."
                                                            " Option: linear/cosine/warmup_cosine")
                        use_ema = gradio.Radio(choices=bool_choices, value=False, label="Use ema model",
                                               info="If set to false, the pt file of the ordinary model will be used. "
                                                    "If true, the pt file of the ema model will be used.")
                    with gradio.Row():
                        num_images = gradio.Slider(minimum=1, maximum=20, step=1, value=2, label="Number of images",
                                                   info="Number of generation images (required). "
                                                        "if class name is `-1` and conditional `is` True, "
                                                        "the model would output one image per class.")
                        class_name = gradio.Slider(minimum=1, maximum=1000, step=1, value=0, label="Class name",
                                                   info="[Note] Enable conditional generation. "
                                                        "If class name is `-1`, "
                                                        "the model would output one image per class. "
                                                        "[Note] The setting range should be [0, num_classes - 1].")
                        cfg_scale = gradio.Slider(minimum=1, maximum=20, step=1, value=3,
                                                  label="Classifier-free guidance",
                                                  info="[Note] Enable conditional generation. "
                                                       "Classifier-free guidance interpolation weight, "
                                                       "users can better generate model effect (recommend)")
                    # Dataset path
                    weight_path = gradio.Textbox(label="Weight path", lines=1,
                                                 placeholder="/your/path/Defect-Diffusion-Model/weights/model.pt",
                                                 info="Your weight model path.")
                    # Saving path
                    result_path = gradio.Textbox(label="Result path", lines=1,
                                                 placeholder="/your/path/Defect-Diffusion-Model/results",
                                                 info="Saving path (required).")

                    btn_generate = gradio.Button("Start generate")
                with gradio.Column():
                    output_images = gradio.Image(label="Result of generate images")
                generate_args = [generate_name, sample, image_format, use_ema, num_images, class_name,
                                 cfg_scale, weight_path, result_path]
                # Generate
                btn_generate.click(fn=gradio_webui.generate, inputs=generate_args, outputs=output_images)

    # Open in browser
    webbrowser.open("http://127.0.0.1:8888")
    webui.queue(max_size=20).launch(server_name="127.0.0.1", server_port=8888)


if __name__ == "__main__":
    main()
