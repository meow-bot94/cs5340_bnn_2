import socket
import logging.handlers
from pathlib import Path
import os

from src.utils.path_getter import PathGetter


class ProcessAwareTimedRotatingFileHandler(logging.handlers.TimedRotatingFileHandler):
    def __init__(self, filename, *args, **kwargs):
        custom_file_path = self._get_logfile_path(filename)
        super().__init__(custom_file_path, *args, **kwargs)

    def _get_logfile_path(self, filename: str) -> str:
        filename = Path(filename)
        current_pid = os.getpid()
        custom_filename = f'{current_pid}_{filename.name}'

        file_parent = PathGetter.get_log_directory()
        file_parent.mkdir(exist_ok=True, parents=True)
        custom_file_path = file_parent / custom_filename

        return custom_file_path.as_posix()
