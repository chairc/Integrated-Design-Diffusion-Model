#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Date   : 2023/11/8 22:44
    @Author : chairc
    @Site   : https://github.com/chairc
"""
import random

import torch
import torch.nn as nn
from torch.cuda.amp import autocast


class BaseNet(nn.Module):
    """
    Base Network
    """

    def __init__(self, in_channel=3, out_channel=3, channel=None, time_channel=256, num_classes=None, image_size=64,
                 device="cpu", act="silu"):
        """
        Initialize the Base network
        :param in_channel: Input channel
        :param out_channel: Output channel
        :param channel: The list of channel
        :param time_channel: Time channel
        :param num_classes: Number of classes
        :param image_size: Adaptive image size
        :param device: Device type
        :param act: Activation function
        """
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.channel = None
        self.init_channel(channel)
        self.time_channel = time_channel
        self.num_classes = num_classes
        self.image_size = image_size
        self.device = device
        self.act = act

        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_embeddings=self.num_classes, embedding_dim=self.time_channel)

    def init_channel(self, channel):
        """
        Init channel
        If channel is None, this function would set a default channel.
        :param channel: Channel
        :return: global self.channel
        """
        if channel is None:
            self.channel = [32, 64, 128, 256, 512, 1024]
        else:
            self.channel = channel

    def pos_encoding(self, time, channels):
        """
        Base network position encoding
        :param time: Time
        :param channels: Channels
        :return: pos_enc
        """
        inv_freq = 1.0 / (10000 ** (torch.arange(start=0, end=channels, step=2, device=self.device).float() / channels))
        inv_freq_value = time.repeat(1, channels // 2) * inv_freq
        pos_enc_a = torch.sin(input=inv_freq_value)
        pos_enc_b = torch.cos(input=inv_freq_value)
        pos_enc = torch.cat(tensors=[pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def fit_one_epoch(self, epoch, images, labels, i, len_dataloader, diffusion, conditional, model, ema_model,
                      optimizer, scaler, mse, fp16, ema, tb_logger, pbar):
        """
        Base train one epoch function (default), subclass models can override this function
        :param epoch: Current epoch
        :param images: Images
        :param labels: Labels
        :param i: Current iter
        :param len_dataloader: Length of dataloader
        :param diffusion: Which sample
        :param conditional: Enable conditional training
        :param model: Current model
        :param ema_model: Ema model
        :param optimizer: Optimizer
        :param scaler: scaler for harf-precision
        :param mse: Mse loss
        :param fp16: Enable harf-precision training
        :param ema: Update ema model
        :param tb_logger: Tensorboard logger
        :param pbar: tqdm bar
        :return: None
        """
        # The images are all resized in dataloader
        images = images.to(self.device)
        # Generates a tensor of size images.shape[0] * images.shape[0] randomly sampled time steps
        time = diffusion.sample_time_steps(images.shape[0]).to(self.device)
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
                    labels = labels.to(self.device)
                    # Random unlabeled hard training, using only time steps and no class information
                    if random.random() < 0.1:
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
                labels = labels.to(self.device)
                # Random unlabeled hard training, using only time steps and no class information
                if random.random() < 0.1:
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
        tb_logger.add_scalar(tag=f"[{self.device}]: MSE", scalar_value=loss.item(),
                             global_step=epoch * len_dataloader + i)

        return model, ema_model, optimizer, scaler
