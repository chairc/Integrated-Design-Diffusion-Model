#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Date   : 2024/3/12 15:30
    @Author : chairc
    @Site   : https://github.com/chairc
"""
import logging
import coloredlogs

logger = logging.getLogger(__name__)
coloredlogs.install(level="INFO")

# IDDM version
__version__ = ["0.0.1", "1.0.0", "1.0.1", "1.0.2", "1.1.0-stable", "1.1.1", "1.1.2-stable", "1.1.3", "1.1.4", "1.1.5",
               "1.1.6", "1.1.7", "1.1.8", "1.1.9", "1.2.0", "1.2.1", "1.2.2", "1.2.3"]


def get_versions():
    """
    Get version list.
    :return: version_list
    """
    version_list = __version__
    logger.info(msg=f"[Note]: Version list is {version_list}")
    return version_list


def get_latest_version():
    """
    Get latest/current version.
    :return: current_version
    """
    current_version = __version__[-1]
    logger.info(msg=f"[Note]: Current version is {current_version}")
    return current_version


def get_old_versions():
    """
    Get old version list.
    :return: old_version_list
    """
    old_version_list = __version__[:-1]
    logger.info(msg=f"[Note]: Old version list is {old_version_list}")
    return old_version_list


def check_version_is_latest(current_version):
    """
    Check if version is latest.
    :param current_version: Current version
    :return: boolean
    """
    if current_version == get_latest_version():
        return True
    return False


def get_version_banner():
    """
    Get version banner.
    """
    print(" _____                   _\n"
          "|  __ \                 (_)\n"
          "| |__) |   _ _ __  _ __  _ _ __   __ _\n"
          "|  _  / | | | '_ \| '_ \| | '_ \ / _` |\n"
          "| | \ \ |_| | | | | | | | | | | | (_| |  _   _   _\n"
          "|_|  \_\__,_|_| |_|_| |_|_|_| |_|\__, | (_) (_) (_)\n"
          "                  __/ |\n"
          "                  |___/\n"
          "                      Are you OK?\n"
          )
    print(f"===============IDDM version: {get_latest_version()}===============\n"
          "Project Author : chairc\n"
          "Project GitHub : https://github.com/chairc/Integrated-Design-Diffusion-Model")


if __name__ == "__main__":
    get_versions()
    get_latest_version()
    get_old_versions()
    get_version_banner()
