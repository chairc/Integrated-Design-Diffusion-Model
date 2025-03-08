#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Date   : 2024/5/4 23:36
    @Author : chairc
    @Site   : https://github.com/chairc
"""
import os
import sys
import argparse
import logging

import coloredlogs

from pytorch_fid.fid_score import save_fid_stats, calculate_fid_given_paths
from pytorch_fid.inception import InceptionV3

sys.path.append(os.path.dirname(sys.path[0]))
from iddm.config.version import get_version_banner
from iddm.utils.initializer import device_initializer

logger = logging.getLogger(__name__)
coloredlogs.install(level="INFO")


def main(args):
    logger.info(msg=f"[Note]: Input params: {args}")
    device_id = args.use_gpu
    paths = args.path
    batch_size = args.batch_size
    num_workers = args.num_workers
    dims = args.dims
    device = device_initializer(device_id=device_id)
    # TODO: Check image size
    # Compute fid
    if args.save_stats:
        save_fid_stats(paths=paths, batch_size=batch_size, device=device, dims=dims, num_workers=num_workers)
        return

    fid_value = calculate_fid_given_paths(paths=paths, batch_size=batch_size, device=device, dims=dims,
                                          num_workers=num_workers)

    logger.info(msg=f"The result of FID: {fid_value}")


if __name__ == "__main__":
    # Before calculating
    # [Note]: We recommend resizing both sets of images to the same format, the same size, and the same number
    parser = argparse.ArgumentParser()
    # Function1: Generated image folder and dataset image folder
    # Function2: Save stats input path and output path (use `--save_stats`)
    parser.add_argument("path", type=str, nargs="*",
                        default=["/your/generated/image/folder/or/stats/input/path",
                                 "/your/dataset/image/folder/or/stats/output/path"],
                        help="Paths to the generated images or to .npz statistic files")
    # Batch size
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for calculation.")
    # Number of workers
    parser.add_argument("--num-workers", type=int, default=0)
    # Dimensionality of Inception features to use
    # Option: 64/192/768/2048
    parser.add_argument("--dims", type=int, default=2048,
                        choices=list(InceptionV3.BLOCK_INDEX_BY_DIM),
                        help="Dimensionality of Inception features to use. By default, uses pool3 features")
    parser.add_argument("--save_stats", action="store_true",
                        help="Generate an npz archive from a directory of samples. "
                             "The first path is used as input and the second as output.")
    # Set the use GPU in normal training (required)
    parser.add_argument("--use_gpu", type=int, default=0)
    args = parser.parse_args()
    # Get version banner
    get_version_banner()
    main(args)
