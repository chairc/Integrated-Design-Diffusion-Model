#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Date   : 2024/1/21 21:46
    @Author : chairc
    @Site   : https://github.com/chairc
"""
import os
import time
import logging
import coloredlogs

from iddm.utils.utils import check_and_create_dir


class CustomLogger(logging.Logger):
    """
    Custom log
    """

    def __init__(self, name, level=logging.INFO, is_webui=False, is_save_log=False, log_path=None):
        super().__init__(name, level)
        self.is_webui = is_webui
        self.webui_text = ""
        self.is_save_log = is_save_log
        self.log_path = log_path

        if self.is_save_log:
            # Set log format
            create_time = time.strftime("%Y%m%d%H%M%S", time.localtime(time.time()))

            if self.log_path is not None:
                # Log file
                log_save_path = os.path.join(self.log_path, "logs")
                check_and_create_dir(log_save_path)
                log_handler = logging.FileHandler(os.path.join(log_save_path, f"{create_time}.log"))
                self.addHandler(log_handler)
            else:
                self.warning("[Warn]: Log path is none.")

            # Console output
            # console_handler = logging.StreamHandler()

            # Add handler
            # self.addHandler(console_handler)

        # Install coloredlogs
        coloredlogs.install(level="INFO", logger=self)

    def debug(self, msg, *args, **kwargs):
        """
        Override debug method
        """
        super().debug(msg, *args, **kwargs)
        if self.is_webui:
            self.webui_text += str(msg) + "\n"
            return self.webui_text

    def info(self, msg, *args, **kwargs):
        """
        Override info method
        """
        super().info(msg, *args, **kwargs)
        if self.is_webui:
            self.webui_text += str(msg) + "\n"
            return self.webui_text

    def warning(self, msg, *args, **kwargs):
        """
        Override warning method
        """
        super().warning(msg, *args, **kwargs)
        if self.is_webui:
            self.webui_text += str(msg) + "\n"
            return self.webui_text

    def error(self, msg, *args, **kwargs):
        """
        Override error method
        """
        super().error(msg, *args, **kwargs)
        if self.is_webui:
            self.webui_text += str(msg) + "\n"
            return self.webui_text
