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
import time

import coloredlogs
import requests
import torch
import torchvision

from PIL import Image
from matplotlib import pyplot as plt
from tqdm import tqdm

from iddm.config.model_list import pretrain_model_choices
from iddm.config.setting import DOWNLOAD_FILE_TEMP_PATH
from iddm.utils.check import check_url

logger = logging.getLogger(__name__)
coloredlogs.install(level="INFO")


def plot_images(images, fig_size=(64, 64)):
    """
    Draw images
    :param images: Image
    :param fig_size: Draw image size
    :return: None
    """
    plt.figure(figsize=fig_size)
    plt.imshow(X=torch.cat([torch.cat([i for i in images.cpu()], dim=-1), ], dim=-2).permute(1, 2, 0).cpu())
    plt.show()


def plot_one_image_in_images(images, fig_size=(64, 64)):
    """
    Draw one image in images
    :param images: Image
    :param fig_size: Draw image size
    :return: None
    """
    plt.figure(figsize=fig_size)
    for i in images.cpu():
        plt.imshow(X=i)
        plt.show()


def save_images(images, path, **kwargs):
    """
    Save images
    :param images: Image
    :param path: Save path
    :param kwargs: Other parameters
    :return: None
    """
    grid = torchvision.utils.make_grid(tensor=images, **kwargs)
    image_array = grid.permute(1, 2, 0).to("cpu").numpy()
    im = Image.fromarray(obj=image_array)
    im.save(fp=path)


def save_one_image_in_images(images, path, generate_name, image_size=None, image_format="jpg", **kwargs):
    """
    Save one image in images
    :param images: Image
    :param generate_name: generate image name
    :param path: Save path
    :param image_size: Resize image size
    :param image_format: Format of the output image
    :param kwargs: Other parameters
    :return: None
    """
    # This is counter
    count = 0
    # Show image in images
    for i in images.cpu():
        grid = torchvision.utils.make_grid(tensor=i, **kwargs)
        image_array = grid.permute(1, 2, 0).to("cpu").numpy()
        im = Image.fromarray(obj=image_array)
        # Rename every images
        im.save(fp=os.path.join(path, f"{generate_name}_{count}.{image_format}"))
        if image_size is not None:
            logger.info(msg=f"Image is resizing {image_size}.")
            # Resize
            # TODO: Super-resolution algorithm replacement
            im = im.resize(size=image_size, resample=Image.LANCZOS)
            im.save(fp=os.path.join(path, f"{generate_name}_{image_size}_{count}.{image_format}"))
        count += 1


def setup_logging(save_path, run_name):
    """
    Set log saving path
    :param save_path: Saving path
    :param run_name: Saving name
    :return: List of file paths
    """
    results_root_dir = save_path
    results_dir = os.path.join(save_path, run_name)
    results_vis_dir = os.path.join(save_path, run_name, "vis")
    results_tb_dir = os.path.join(save_path, run_name, "tensorboard")
    # Root folder
    os.makedirs(name=results_root_dir, exist_ok=True)
    # Saving folder
    os.makedirs(name=results_dir, exist_ok=True)
    # Visualization folder
    os.makedirs(name=results_vis_dir, exist_ok=True)
    # Visualization folder for Tensorboard
    os.makedirs(name=results_tb_dir, exist_ok=True)
    return [results_root_dir, results_dir, results_vis_dir, results_tb_dir]


def delete_files(path):
    """
    Clear files
    :param path: Path
    :return: None
    """
    if os.path.exists(path):
        if os.path.isfile(path):
            os.remove(path)
        else:
            shutil.rmtree(path=path)
        logger.info(msg=f"Folder '{path}' deleted.")
    else:
        logger.warning(msg=f"Folder '{path}' does not exist.")


def save_train_logging(arg, save_path):
    """
    Save train log
    :param arg: Argparse
    :param save_path: Save path
    :return: None
    """
    with open(file=f"{save_path}/train.log", mode="a") as f:
        current_time = time.strftime("%H:%M:%S", time.localtime())
        f.write(f"{current_time}: {arg}\n")
    f.close()


def check_and_create_dir(path):
    """
    Check and create not exist folder
    :param path: Create path
    :return: None
    """
    logger.info(msg=f"Check and create folder '{path}'.")
    os.makedirs(name=path, exist_ok=True)


def download_files(url_list=None, save_path=None):
    """
    Downloads files
    :param url_list: url list
    :param save_path: Save path
    """
    # Temp download path
    if save_path is None:
        download_file_temp_path = DOWNLOAD_FILE_TEMP_PATH
    else:
        download_file_temp_path = save_path
    # Check and create
    check_and_create_dir(path=download_file_temp_path)
    # Check url list
    for url in url_list:
        logger.info(msg=f"Current download url is {url}")
        file_name = check_url(url=url)

        # Send the request with stream=True
        with requests.get(url, stream=True) as response:
            # Check for HTTP errors
            response.raise_for_status()
            # Get the total size of the file (if possible)
            total_size_in_bytes = int(response.headers.get("content-length", 0))
            block_size = 1024  # 1KB for each read
            progress_bar = tqdm(total=total_size_in_bytes, unit="B", unit_scale=True, desc=f"Downloading {file_name}")

            # Open the file in binary write mode
            with open(os.path.join(download_file_temp_path, file_name), "wb") as file:
                for data in response.iter_content(block_size):
                    # Write the data to the file
                    file.write(data)
                    # Update the progress bar
                    progress_bar.update(len(data))
            # Close the progress bar
            progress_bar.close()
            logger.info(msg=f"Current {url} is download successfully.")
    logger.info(msg="Everything is downloaded.")


def download_model_pretrain_model(pretrain_type="df", network="unet", image_size=64, **kwargs):
    """
    Download pre-trained model in GitHub repository
    :param pretrain_type: Type of pre-trained model
    :param network: Network
    :param image_size: Image size
    :param kwargs: Other parameters
    :return new_pretrain_path
    """
    # Check image size
    if isinstance(image_size, int):
        image_size = str(image_size)
    else:
        raise ValueError("Official pretrain model's image size must be int, such as 64 or 120.")
    # Download diffusion pretrain model
    if pretrain_type == "df":
        df_type = kwargs.get("df_type", "default")
        conditional_type = "conditional" if kwargs.get("df_type", True) else "unconditional"
        # Download pretrain model
        if df_type == "default":
            pretrain_model_url = pretrain_model_choices[pretrain_type][df_type][network][conditional_type][image_size]
        # Download sample model.
        # If use cifar-10 dataset, you can set cifar10 pretrain model
        elif df_type == "exp":
            model_name = kwargs.get("model_name", "cifar10")
            pretrain_model_url = pretrain_model_choices[pretrain_type][df_type][network][conditional_type][image_size][
                model_name]
        else:
            raise TypeError(f"Diffusion model type '{df_type}' is not supported.")
    # Download super resolution pretrain model
    elif pretrain_type == "sr":
        act = kwargs.get("act", "silu")
        pretrain_model_url = pretrain_model_choices[pretrain_type][network][act][image_size]
    else:
        raise TypeError(f"Pretrain type '{pretrain_type}' is not supported.")

    # Download model
    download_files(url_list=[pretrain_model_url])
    logger.info(msg=f"Current pretrain model path '{pretrain_model_url}' is download successfully.")
    # Get file name
    parts = pretrain_model_url.split("/")
    filename = parts[-1]
    new_pretrain_path = os.path.join(DOWNLOAD_FILE_TEMP_PATH, filename)
    return new_pretrain_path
