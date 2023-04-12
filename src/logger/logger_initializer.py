import datetime
import logging
import logging.config
import os
from pathlib import Path
from typing import Any, Dict

import pytz

from src.utils.monostate import Monostate
from src.utils.path_getter import PathGetter
from src.utils.yaml_reader import YamlReader


class LoggerInitializer(Monostate):

    @property
    def monostate_defaults(self) -> Dict[str, Any]:
        return {
            'has_init': False,
        }

    @property
    def logger_config_path(self) -> Path:
        return PathGetter.get_config_directory() / 'logger_config.yaml'

    @property
    def timezone(self) -> str:
        return 'Asia/Singapore'

    @property
    def tzinfo(self) -> pytz.timezone:
        return pytz.timezone(self.timezone)

    def _get_default_config(self) -> dict:
        return YamlReader.read_yaml_config(self.logger_config_path)

    def _init_logger(self):
        logging_config_dict = self._get_default_config()
        logging.Formatter.converter = lambda *x: datetime.datetime.now(self.tzinfo).timetuple()
        logging.config.dictConfig(logging_config_dict)

    def init(self):
        if not self.has_init:
            self._init_logger()
            pid = os.getpid()
            logging.info(f'Current process ID: {pid}')
            self.has_init = True
        return self.has_init
