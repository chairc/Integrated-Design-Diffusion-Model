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
from collections import OrderedDict

sys.path.append(os.path.dirname(sys.path[0]))
from model.ddpm import Diffusion
from model.modules import EMA
from model.network import UNet
from utils.initializer import device_initializer, seed_initializer
from utils.lr_scheduler import set_cosine_lr
from utils.utils import plot_images, save_images, get_dataset, setup_logging

logger = logging.getLogger(__name__)
coloredlogs.install(level="INFO")


def train(rank=None, args=None):
    """
    训练
    :param args: 输入参数
    :return: None
    """
    logger.info(msg=f"[{rank}]: Input params: {args}")
    # 初始化种子
    seed_initializer(seed_id=args.seed)
    # 运行名称
    run_name = args.run_name
    # 输入图像大小
    image_size = args.image_size
    # 优化器选择
    optim = args.optim
    # 学习率大小
    init_lr = args.lr
    # 学习率方法
    lr_func = args.lr_func
    # 类别个数
    num_classes = args.num_classes
    # classifier-free guidance插值权重，用户更好生成模型效果
    cfg_scale = args.cfg_scale
    # 是否开启条件训练
    conditional = args.conditional
    # 初始化保存模型标识位，这里检测是否单卡训练还是多卡训练
    save_models = True
    # 是否开启分布式训练
    if args.distributed and torch.cuda.device_count() > 1 and torch.cuda.is_available():
        distributed = True
        world_size = args.world_size
        # 设置地址和端口
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12345"
        # 进程总数等于显卡数量
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo", rank=rank,
                                world_size=world_size)
        # 设置设备ID
        device = torch.device("cuda", rank)
        # 可能出现随机性错误，使用可减少cudnn随机性错误
        # torch.backends.cudnn.deterministic = True
        # 同步
        dist.barrier()
        # 如果分布式训练是第一块显卡，则保存模型标识位为真
        if dist.get_rank() != args.main_gpu:
            save_models = False
        logger.info(msg=f"[{device}]: Successfully Use distributed training.")
    else:
        distributed = False
        # 运行设备
        device = device_initializer()
        logger.info(msg=f"[{device}]: Successfully Use normal training.")
    # 是否开启半精度训练
    fp16 = args.fp16
    # 保存模型周期
    save_model_interval = args.save_model_interval
    # 开始保存模型周期
    start_model_interval = args.start_model_interval
    # 开启可视化数据
    vis = args.vis
    # 保存路径
    result_path = args.result_path
    # 创建训练生成日志
    results_logging = setup_logging(save_path=result_path, run_name=run_name)
    results_dir = results_logging[1]
    results_vis_dir = results_logging[2]
    results_tb_dir = results_logging[3]
    # 数据集加载器
    dataloader = get_dataset(args=args, distributed=distributed)
    # 恢复训练
    resume = args.resume
    # 模型
    if not conditional:
        model = UNet(device=device, image_size=image_size).to(device)
    else:
        model = UNet(num_classes=num_classes, device=device, image_size=image_size).to(device)
    # 分布式训练
    if distributed:
        model = nn.parallel.DistributedDataParallel(module=model, device_ids=[device], find_unused_parameters=True)
    # 模型优化器
    if optim == "adam":
        optimizer = torch.optim.Adam(params=model.parameters(), lr=init_lr)
    else:
        optimizer = torch.optim.AdamW(params=model.parameters(), lr=init_lr)
    # 恢复训练
    if resume:
        load_model_dir = args.load_model_dir
        start_epoch = args.start_epoch
        # 加载上一个模型
        load_epoch = str(start_epoch - 1).zfill(3)
        model_path = os.path.join(result_path, load_model_dir, f"model_{load_epoch}.pt")
        optim_path = os.path.join(result_path, load_model_dir, f"optim_model_{load_epoch}.pt")
        model_dict = model.state_dict()
        model_weights_dict = torch.load(f=model_path, map_location=device)
        model_weights_dict = {k: v for k, v in model_weights_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
        model_dict.update(model_weights_dict)
        model.load_state_dict(state_dict=OrderedDict(model_dict))
        logger.info(msg=f"[{device}]: Successfully load model model_{load_epoch}.pt")
        # 加载优化器参数
        optim_weights_dict = torch.load(f=optim_path, map_location=device)
        optimizer.load_state_dict(state_dict=optim_weights_dict)
        logger.info(msg=f"[{device}]: Successfully load optimizer optim_model_{load_epoch}.pt")
    else:
        start_epoch = 0
    if fp16:
        logger.info(msg=f"[{device}]: Fp16 training is opened.")
        # 用于缩放梯度，以防止溢出
        scaler = GradScaler()
    else:
        logger.info(msg=f"[{device}]: Fp32 training.")
        scaler = None
    # 损失函数
    mse = nn.MSELoss()
    # 初始化扩散模型
    diffusion = Diffusion(img_size=image_size, device=device)
    # 日志记录器
    tb_logger = SummaryWriter(log_dir=results_tb_dir)
    # 数据加载器中数据集批次个数
    len_dataloader = len(dataloader)
    # EMA指数移动平均对于单类别优势可能不如多类别
    ema = EMA(beta=0.995)
    # EMA模型
    ema_model = copy.deepcopy(model).eval().requires_grad_(False)

    logger.info(msg=f"[{device}]: Start training.")
    # 开始迭代
    for epoch in range(start_epoch, args.epochs):
        logger.info(msg=f"[{device}]: Start epoch {epoch}:")
        # 设置学习率
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
        # 初始化images和labels
        images, labels = None, None
        for i, (images, labels) in enumerate(pbar):
            # 图片均为dataloader中resize后的图
            images = images.to(device)
            # 生成大小为images.shape[0] * images.shape[0]的随机采样时间步长的张量
            time = diffusion.sample_time_steps(images.shape[0]).to(device)
            # 添加噪声，返回为t时刻的x值和标准正态分布
            x_time, noise = diffusion.noise_images(x=images, time=time)
            # 开启半精度训练
            if fp16:
                # 使用半精度
                with autocast():
                    # 半精度无条件训练
                    if not conditional:
                        # 半精度无条件模型预测
                        predicted_noise = model(x_time, time)
                    # 有条件训练，需要加入标签
                    else:
                        labels = labels.to(device)
                        # 随机进行无标签困难训练，只使用时间步长不使用类别信息
                        if np.random.random() < 0.1:
                            labels = None
                        # 半精度有条件模型预测
                        predicted_noise = model(x_time, time, labels)
                    # 计算MSE损失，需要使用x在t时刻的标准正态分布和预测后的噪声进行损失计算
                    loss = mse(noise, predicted_noise)
                # 优化器清零模型参数的梯度
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            # 使用全精度
            else:
                # 全精度无条件训练
                if not conditional:
                    # 全精度无条件模型预测
                    predicted_noise = model(x_time, time)
                # 全精度有条件训练，需要加入标签
                else:
                    labels = labels.to(device)
                    # 随机进行无标签困难训练，只使用时间步长不使用类别信息
                    if np.random.random() < 0.1:
                        labels = None
                    # 全精度有条件模型预测
                    predicted_noise = model(x_time, time, labels)
                # 计算MSE损失，需要使用x在t时刻的标准正态分布和预测后的噪声进行损失计算
                loss = mse(noise, predicted_noise)
                # 优化器清零模型参数的梯度
                optimizer.zero_grad()
                # 自动计算梯度
                loss.backward()
                # 优化器更新模型的参数
                optimizer.step()
            # EMA
            ema.step_ema(ema_model=ema_model, model=model)

            # TensorBoard记录日志
            pbar.set_postfix(MSE=loss.item())
            tb_logger.add_scalar(tag=f"[{device}]: MSE", scalar_value=loss.item(),
                                 global_step=epoch * len_dataloader + i)

        # 分布式在训练过程中进行同步
        if distributed:
            dist.barrier()

        # 主进程中保存和验证模型
        if save_models:
            # 保存模型
            save_name = f"model_{str(epoch).zfill(3)}"
            if not conditional:
                # 保存pt文件
                torch.save(obj=model.state_dict(), f=os.path.join(results_dir, f"model_last.pt"))
                torch.save(obj=optimizer.state_dict(), f=os.path.join(results_dir, f"optim_last.pt"))
                # 开启可视化
                if vis:
                    # images.shape[0]为当前这个批次的图像个数
                    sampled_images = diffusion.sample(model=model, n=images.shape[0])
                    save_images(images=sampled_images, path=os.path.join(results_vis_dir, f"{save_name}.jpg"))
                # 周期保存pt文件
                if save_model_interval and epoch > start_model_interval:
                    torch.save(obj=model.state_dict(), f=os.path.join(results_dir, f"{save_name}.pt"))
                    torch.save(obj=optimizer.state_dict(), f=os.path.join(results_dir, f"optim_{save_name}.pt"))
                    logger.info(msg=f"Save the {save_name}.pt, and optim_{save_name}.pt.")
                logger.info(msg="Save the model.")
            else:
                # 保存文件
                torch.save(obj=model.state_dict(), f=os.path.join(results_dir, f"model_last.pt"))
                torch.save(obj=ema_model.state_dict(), f=os.path.join(results_dir, f"ema_model_last.pt"))
                torch.save(obj=optimizer.state_dict(), f=os.path.join(results_dir, f"optim_last.pt"))
                # 开启可视化
                if vis:
                    labels = torch.arange(num_classes).long().to(device)
                    sampled_images = diffusion.sample(model=model, n=len(labels), labels=labels, cfg_scale=cfg_scale)
                    ema_sampled_images = diffusion.sample(model=ema_model, n=len(labels), labels=labels,
                                                          cfg_scale=cfg_scale)
                    plot_images(images=sampled_images)
                    save_images(images=sampled_images, path=os.path.join(results_vis_dir, f"{save_name}.jpg"))
                    save_images(images=ema_sampled_images, path=os.path.join(results_vis_dir, f"{save_name}_ema.jpg"))
                if save_model_interval and epoch > start_model_interval:
                    torch.save(obj=model.state_dict(), f=os.path.join(results_dir, f"{save_name}.pt"))
                    torch.save(obj=ema_model.state_dict(), f=os.path.join(results_dir, f"ema_{save_name}.pt"))
                    torch.save(obj=optimizer.state_dict(), f=os.path.join(results_dir, f"optim_{save_name}.pt"))
                    logger.info(msg=f"Save the {save_name}.pt, ema_{save_name}.pt, and optim_{save_name}.pt.")
                logger.info(msg="Save the model.")
        logger.info(msg=f"[{device}]: Finish epoch {epoch}:")
    logger.info(msg=f"[{device}]: Finish training.")

    # 清理分布式环境
    if distributed:
        dist.destroy_process_group()


def main(args):
    if args.distributed:
        gpus = torch.cuda.device_count()
        mp.spawn(train, args=(args,), nprocs=gpus)
    else:
        train(args=args)


if __name__ == "__main__":
    # 训练模型参数
    parser = argparse.ArgumentParser()
    # 设置初始化种子（必须设置）
    parser.add_argument("--seed", type=int, default=0)
    # 开启条件训练（必须设置）
    # 若开启可修改自定义配置，详情参考最下面分界线
    parser.add_argument("--conditional", type=bool, default=False)
    # 初始化模型的文件名称（必须设置）
    parser.add_argument("--run_name", type=str, default="df")
    # 训练总迭代次数（必须设置）
    parser.add_argument("--epochs", type=int, default=100)
    # 训练批次大小（必须设置）
    parser.add_argument("--batch_size", type=int, default=1)
    # 用于数据加载的子进程数量（酌情设置）
    # 大量占用CPU和内存，但可以加快训练速度
    parser.add_argument("--num_workers", type=int, default=0)
    # 输入图像大小（必须设置）
    parser.add_argument("--image_size", type=int, default=64)
    # 数据集路径（必须设置）
    # 有条件数据集，例如cifar10，每个类别一个文件夹，路径为主文件夹
    # 无条件数据集，所有图放在一个文件夹，路径为图片文件夹
    parser.add_argument("--dataset_path", type=str, default="/your/path/Defect-Diffusion-Model/datasets/dir")
    # 开启半精度训练（酌情设置）
    # 有效减少显存使用，但无法保证训练精度和训练结果
    parser.add_argument("--fp16", type=bool, default=False)
    # 优化器选择，adam/adamw（酌情设置）
    parser.add_argument("--optim", type=str, default="adamw")
    # 学习率（酌情设置）
    parser.add_argument("--lr", type=int, default=3e-4)
    # 学习率方法（酌情设置）
    # 不设置时为空，可设置cosine，warmup_cosine
    parser.add_argument("--lr_func", type=str, default="")
    # 保存路径（必须设置）
    parser.add_argument("--result_path", type=str, default="/your/path/Defect-Diffusion-Model/results")
    # 是否每次训练储存（建议设置）
    # 根据可视化生成样本信息筛选模型
    parser.add_argument("--save_model_interval", type=bool, default=True)
    # 设置开始每次训练存储的epoch编号（酌情设置）
    # 该设置可节约磁盘空间，若不设置默认-1，若设置则从第epoch时开始保存每次训练pt文件，需要与--save_model_interval同时开启
    parser.add_argument("--start_model_interval", type=int, default=-1)
    # 打开可视化数据集信息，根据可视化生成样本信息筛选模型（建议设置）
    parser.add_argument("--vis", type=bool, default=True)
    # 训练异常中断（酌情设置）
    # 1.恢复训练将设置为“True” 2.设置异常中断的epoch编号 3.写入中断的epoch上一个加载模型的所在文件夹
    # 注意：设置异常中断的epoch编号若在--start_model_interval参数条件外，则不生效
    # 例如开始保存模型时间为100，中断编号为50，由于我们没有保存模型，所以无法设置任意加载epoch点
    # 每次训练我们都会保存xxx_last.pt文件，所以我们需要使用最后一次保存的模型进行中断训练
    parser.add_argument("--resume", type=bool, default=False)
    parser.add_argument("--start_epoch", type=int, default=-1)
    parser.add_argument("--load_model_dir", type=str, default="")

    # ======================================开启分布式训练分界线======================================
    # 开启分布式训练（酌情设置）
    parser.add_argument("--distributed", type=bool, default=True)
    # 设置分布式中主显卡（必须设置）
    # 默认为0
    parser.add_argument('--main_gpu', type=int, default=0)
    # 分布式训练的节点等级node rank（酌情设置）
    parser.add_argument('--world_size', type=int, default=2)

    # ==========================开启条件生成分界线（若设置--conditional为True设置这里）==========================
    # 类别个数（必须设置）
    parser.add_argument("--num_classes", type=int, default=1)
    # classifier-free guidance插值权重，用户更好生成模型效果（建议设置）
    parser.add_argument("--cfg_scale", type=int, default=3)

    args = parser.parse_args()

    main(args)
