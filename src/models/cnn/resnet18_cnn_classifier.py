from typing import Type

from torch import nn

from src.common.cnn_image_classifier import CnnImageClassifier
from src.common.model_creator import ModelCreator
from src.model_creation.resnet18_cnn_creator import Resnet18CnnCreator


class Resnet18CnnClassifier(CnnImageClassifier):
    @property
    def name(self) -> str:
        return 'resnet18cnn'

    def _get_loss_criterion(self):
        return nn.CrossEntropyLoss()

    @property
    def model_creator_class(self) -> Type[ModelCreator]:
        return Resnet18CnnCreator

    def unfreeze_fc_layer(self):
        self.unfreeze_layer('fc')

    def freeze_all_layers_except_fc(self):
        self.freeze_all_layers()
        self.unfreeze_fc_layer()
