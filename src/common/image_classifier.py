import os
import uuid
from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple, Type

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.common.model_creator import ModelCreator
from src.data_loader.dataset import Dataset
from src.scoring.class_scorer import ClassScorer
from src.utils.path_getter import PathGetter


class ImageClassifier(ABC):
    def __init__(self, dataset: Dataset, device: str = None):
        self._device = self._infer_device(device)
        self._dataset = dataset
        self._model = None
        self._uuid = self._get_unique_id()
        self._best_results = dict()

    def _get_unique_id(self):
        return uuid.uuid4().urn[9:]

    def _infer_device(self, device: Optional[str]):
        if device:
            return device
        return 'cuda' if torch.cuda.is_available() else 'cpu'

    @property
    @abstractmethod
    def name(self) -> str:
        return ''

    @property
    @abstractmethod
    def model_creator_class(self) -> Type[ModelCreator]:
        pass

    def _init(self):
        self._model = self.model_creator_class(
            self._dataset, self._device
        ).create()

    def init(self):
        self._init()
        return self

    @abstractmethod
    def _get_loss_criterion(self):
        pass

    @abstractmethod
    def _fit(self, num_epoch: int, verbose) -> Tuple[pd.DataFrame, Dict]:
        pass

    def fit(self, num_epoch: int, verbose=True) -> Tuple[pd.DataFrame, Dict]:
        return self._fit(num_epoch, verbose)

    @abstractmethod
    def predict(self, dataloader: DataLoader):
        pass

    def score(self, dataloader: DataLoader):
        y_true, y_pred = self.predict(dataloader)
        return ClassScorer.score(y_true, y_pred)

    def save_model(self, result_dict: Dict):
        process_id = os.getpid()
        file_name = f'{self.name}_{process_id}_{self._uuid}.pt'
        storage_dir = PathGetter.get_assets_directory() / 'model_weights'
        file_path = storage_dir / file_name
        torch.save(result_dict, file_path)

    @abstractmethod
    def _load_model(self, *args, **kwargs):
        pass

    def load_model(self, *args, **kwargs):
        if self._model is None:
            self._init()
        self._load_model(*args, **kwargs)
        return self
