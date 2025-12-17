#!/usr/bin/env python
# -*- coding:utf-8 -*-

# Copyright 2025 IDDM Authors

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
    @Date   : 2024/1/21 21:46
    @Author : chairc
    @Site   : https://github.com/chairc
"""
import os
import time
import logging
import coloredlogs
from typing import Optional, Union, Dict, Any

# Global log instance cache (isolated by name and type)
_global_loggers = {}

# Global log configuration (stores parameters for init_logger settings)
_global_log_config: Dict[str, Any] = {
    "level": None,
    "is_save_log": None,
    "log_path": None,
    "initialized": False  # Whether the tag is initialized
}


class BaseLogger(logging.Logger):
    """
    Basic log class, which provides general log functions
    """

    def __init__(
            self,
            name: str,
            level: Union[int, str] = logging.INFO,
            is_save_log: bool = False,
            log_path: Optional[str] = None
    ):
        super().__init__(name, level)
        self.is_save_log = is_save_log
        self.log_path = log_path
        self._init_handlers()
        self._setup_colored_logs()

    def _init_handlers(self) -> None:
        """
        Initialize the Log Processor (Console + File)
        """
        # Clear existing processors to avoid duplicate outputs
        self.handlers.clear()

        # Add a console processor
        self._add_console_handler()

        # Add a file handler (if you need to save logs)
        if self.is_save_log and self.log_path:
            self._add_file_handler()

    def _add_console_handler(self) -> None:
        """
        Add a console processor
        """
        console_handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s   %(name)s   %(levelname)s   %(message)s')
        console_handler.setFormatter(formatter)
        self.addHandler(console_handler)

    def _add_file_handler(self) -> None:
        """
        Add a file handler
        """
        create_time = time.strftime("%Y%m%d%H%M%S", time.localtime())
        log_save_path = os.path.join(self.log_path, "logs")
        os.makedirs(name=log_save_path, exist_ok=True)

        # Different types of logs use different file prefixes
        prefix = "webui" if isinstance(self, WebUILogger) else "app"
        log_file = os.path.join(log_save_path, f"{prefix}_{create_time}.log")

        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        formatter = logging.Formatter("%(asctime)s   %(name)s   %(levelname)s   %(message)s")
        file_handler.setFormatter(formatter)
        self.addHandler(file_handler)

    def _setup_colored_logs(self) -> None:
        """
        Configure the color log
        """
        coloredlogs.install(level=self.level, logger=self)

    def refresh_config(self, level: Union[int, str], is_save_log: bool, log_path: Optional[str]):
        """
        Refresh log instance configuration (used to update existing instances after init_logger)
        """
        self.level = level
        self.is_save_log = is_save_log
        self.log_path = log_path
        self._init_handlers()
        self._setup_colored_logs()


class WebUILogger(BaseLogger):
    """
    WebUI log class supports front-end text accumulation
    """

    def __init__(
            self,
            name: str,
            level: Union[int, str] = logging.INFO,
            is_save_log: bool = False,
            log_path: Optional[str] = None
    ):
        super().__init__(name, level, is_save_log, log_path)
        # Cumulative text for WebUI display
        self.webui_text = ""

    def _update_webui_text(self, msg: str) -> str:
        """
        Updated WebUI text cache
        """
        self.webui_text += f"{msg}\n"
        return self.webui_text

    def debug(self, msg: str, *args, **kwargs) -> str:
        """
        Debug level log
        """
        super().debug(msg, *args, **kwargs)
        return self._update_webui_text(msg)

    def info(self, msg: str, *args, **kwargs) -> str:
        """
        Info level log
        """
        super().info(msg, *args, **kwargs)
        return self._update_webui_text(msg)

    def warning(self, msg: str, *args, **kwargs) -> str:
        """
        Warning level log
        """
        super().warning(msg, *args, **kwargs)
        return self._update_webui_text(msg)

    def error(self, msg: str, *args, **kwargs) -> str:
        """
        Error level log
        """
        super().error(msg, *args, **kwargs)
        return self._update_webui_text(msg)

    def critical(self, msg: str, *args, **kwargs) -> str:
        """
        Critical level log
        """
        super().critical(msg, *args, **kwargs)
        return self._update_webui_text(msg)


class AppLogger(BaseLogger):
    """
    Ordinary application logging, focusing on back-end logging
    """

    def __init__(
            self,
            name: str,
            level: Union[int, str] = logging.INFO,
            is_save_log: bool = False,
            log_path: Optional[str] = None
    ):
        super().__init__(name, level, is_save_log, log_path)


def init_logger(
        level: Union[int, str] = logging.INFO,
        is_save_log: bool = False,
        log_path: Optional[str] = None
) -> None:
    """
    Initialize the global log configuration (only the first call takes effect)
    :param level: Log level
    :param is_save_log: Whether to save the log to a file
    :param log_path: Log saving path
    :return: None
    """
    global _global_log_config
    if _global_log_config["initialized"]:
        logging.warning("The global log configuration is initialized, skipping duplicate settings")
        return

    # Store global configurations
    _global_log_config.update({
        "level": level,
        "is_save_log": is_save_log,
        "log_path": log_path,
        "initialized": True
    })

    # Refresh existing log instances (apply new configuration)
    for logger in _global_loggers.values():
        logger.refresh_config(
            level=level,
            is_save_log=is_save_log,
            log_path=log_path
        )


def get_logger(
        name: str,
        logger_type: str = "app",
        level: Union[int, str] = logging.INFO,
        is_save_log: bool = False,
        log_path: Optional[str] = None
) -> Union[AppLogger, WebUILogger]:
    """
    Get a global log instance (singleton mode)
    :param name: Logger name
    :param logger_type: Logger type, options are "app" (application log) and "webui" (WebUI log), default is "app"
    :param level: Log level
    :param is_save_log: Whether to save the log to a file
    :param log_path: Log saving path
    :return: Log instance
    """

    global _global_loggers, _global_log_config

    # The global configuration of the init_logger is used first, and the parameters are overwritten
    level = level or _global_log_config.get("level", logging.INFO)
    is_save_log = is_save_log if is_save_log is not None else _global_log_config.get("is_save_log", False)
    log_path = log_path or _global_log_config.get("log_path")

    # Use (name + type) as the unique key to ensure isolation
    key = f"{name}_{logger_type}"

    if key not in _global_loggers:
        if logger_type == "webui":
            _global_loggers[key] = WebUILogger(
                name=name,
                level=level,
                is_save_log=is_save_log,
                log_path=log_path
            )
        else:
            _global_loggers[key] = AppLogger(
                name=name,
                level=level,
                is_save_log=is_save_log,
                log_path=log_path
            )

    return _global_loggers[key]


# Default application log instance (global can be called directly)
default_logger = get_logger(name="iddm", logger_type="app")
