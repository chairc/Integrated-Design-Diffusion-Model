#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Date   : 2023/6/15 17:12
    @Author : chairc
    @Site   : https://github.com/chairc
"""
import os
import logging
import shutil

import coloredlogs
import torch
import torchvision

from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)
coloredlogs.install(level="INFO")


def plot_images(images, fig_size=(64, 64)):
    """
    绘制图像
    :param images: 图像
    :param fig_size: 绘制大小
    :return: None
    """
    plt.figure(figsize=fig_size)
    plt.imshow(X=torch.cat([torch.cat([i for i in images.cpu()], dim=-1), ], dim=-2).permute(1, 2, 0).cpu())
    plt.show()


def save_images(images, path, **kwargs):
    """
    保存图像
    :param images: 图像
    :param path: 保存地址
    :param kwargs: 其他参数
    :return: None
    """
    grid = torchvision.utils.make_grid(tensor=images, **kwargs)
    image_array = grid.permute(1, 2, 0).to("cpu").numpy()
    im = Image.fromarray(obj=image_array)
    im.save(fp=path)


def get_dataset(args):
    """
    获取数据集

    自动划分标签 torchvision.datasets.ImageFolder(args.dataset_path, transform=transforms)
    若数据集为：
        dataset_path/class_1/image_1.jpg
        dataset_path/class_1/image_2.jpg
        ...
        dataset_path/class_2/image_1.jpg
        dataset_path/class_2/image_2.jpg
        ...
    其中，dataset_path是数据集所在的根目录，class_1, class_2等是数据集中的不同类别，每个类别下包含若干张图像文件。
    使用ImageFolder类可以方便地加载这种文件夹结构的图像数据集，并自动为每个图像分配相应的标签。
    可以通过传递dataset_path参数指定数据集所在的根目录，并通过其他可选参数进行图像预处理、标签转换等操作。

    :param args: 参数
    :return: dataloader
    """
    transforms = torchvision.transforms.Compose([
        # 重新设置大小
        # torchvision.transforms.Resize(80), args.image_size + 1/4 * args.image_size
        torchvision.transforms.Resize(size=int(args.image_size + args.image_size / 4)),
        # 随机调整裁剪
        torchvision.transforms.RandomResizedCrop(size=args.image_size, scale=(0.8, 1.0)),
        # 转为Tensor格式
        torchvision.transforms.ToTensor(),
        # 做标准化处理，均值标准差均为0.5
        torchvision.transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    # 加载当前路径下的文件夹数据，根据每个文件名下的数据集自动划分标签
    dataset = torchvision.datasets.ImageFolder(root=args.dataset_path, transform=transforms)
    dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                            pin_memory=True)
    return dataloader


def setup_logging(save_path, run_name):
    """
    设置日志存储位置
    :param save_path: 保存路径
    :param run_name: 保存名称
    :return: 文件路径List
    """
    results_root_dir = save_path
    results_dir = os.path.join(save_path, run_name)
    results_vis_dir = os.path.join(save_path, run_name, "vis")
    results_tb_dir = os.path.join(save_path, run_name, "tensorboard")
    # 结果主文件夹
    os.makedirs(name=results_root_dir, exist_ok=True)
    # 结果保存文件夹
    os.makedirs(name=results_dir, exist_ok=True)
    # 可视化文件夹
    os.makedirs(name=results_vis_dir, exist_ok=True)
    # 可视化数据文件夹
    os.makedirs(name=results_tb_dir, exist_ok=True)
    return [results_root_dir, results_dir, results_vis_dir, results_tb_dir]


def delete_files(path):
    """
    清除文件
    :param path: 路径
    :return: None
    """
    if os.path.exists(path=path):
        if os.path.isfile(path=path):
            os.remove(path=path)
        else:
            shutil.rmtree(path=path)
        logger.info(msg=f"Folder '{path}' deleted.")
    else:
        logger.warning(msg=f"Folder '{path}' does not exist.")
