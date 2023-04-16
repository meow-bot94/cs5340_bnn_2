from typing import Type

from torch import nn

from src.common.cnn_image_classifier import CnnImageClassifier
from src.common.model_creator import ModelCreator
from src.model_creation.resnet50_cnn_creator import Resnet50CnnCreator


class Resnet50CnnClassifier(CnnImageClassifier):
    @property
    def name(self) -> str:
        return 'resnet50cnn'

    def _get_loss_criterion(self):
        return nn.CrossEntropyLoss()

    @property
    def model_creator_class(self) -> Type[ModelCreator]:
        return Resnet50CnnCreator
