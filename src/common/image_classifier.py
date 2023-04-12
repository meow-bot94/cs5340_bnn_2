from abc import ABC, abstractmethod
from pathlib import Path

import torch

from src.data_loader.dataset import Dataset


class ImageClassifier(ABC):
    def __init__(self, dataset: Dataset, device: str = None):
        self._device = device if device else self._infer_device()
        self._dataset = dataset

    def _infer_device(self):
        return 'cuda' if torch.cuda.is_available() else 'cpu'

    @abstractmethod
    def _init(self):
        pass

    def init(self):
        return self

    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def predict(self):
        pass

    @abstractmethod
    def save_model(self):
        pass

    @abstractmethod
    def load_model(self, file_path: Path):
        pass
