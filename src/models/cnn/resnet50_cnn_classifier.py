from typing import Type

from torch.utils.data import DataLoader

from src.common.image_classifier import ImageClassifier
from src.common.model_creator import ModelCreator
from src.model_creation.resnet50_cnn_creator import Resnet50CnnCreator
from src.prediction.cnn_batch_predictor import CnnBatchPredictor
from src.scoring.class_scorer import ClassScorer
from src.training.cnn_trainer import CnnTrainer


class Resnet50CnnClassifier(ImageClassifier):
    @property
    def name(self) -> str:
        return 'resnet50cnn'

    @property
    def model_creator_class(self) -> Type[ModelCreator]:
        return Resnet50CnnCreator

    def _fit(self, num_epoch: int, verbose):
        assert self._model is not None, 'Model not initiated. Run init first.'
        trainer = CnnTrainer(self._model, self._dataset, self._device)
        return trainer.train(num_epoch, verbose)

    def predict(self, dataloader: DataLoader):
        return CnnBatchPredictor(self._model, dataloader, self._device).predict()

    def score(self, dataloader: DataLoader):
        y_true, y_pred = self.predict(dataloader)
        return ClassScorer.score(y_true, y_pred)

    def _load_model(self, model_state_dict: dict):
        self._model.load_state_dict(model_state_dict)
