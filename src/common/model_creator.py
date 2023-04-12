from abc import ABC, abstractmethod

from src.data_loader.dataset import Dataset


class ModelCreator(ABC):
    def __init__(self, dataset: Dataset, device: str):
        self._dataset = dataset
        self._device = device

    @abstractmethod
    def _create(self, dataset: Dataset, device: str):
        pass

    def create(self):
        return self._create(self._dataset, self._device)
