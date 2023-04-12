from pathlib import Path
from typing import Union

from src.data_loader.dataset import Dataset


class DatasetGetter:
    def __init__(self, root_dir: Union[str, Path], batch_size: int = 12):
        self._root_dir = Path(root_dir)
        self._batch_size = batch_size

    def get(self):
        return Dataset(self._root_dir, self._batch_size)
