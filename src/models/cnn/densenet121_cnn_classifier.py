from typing import Type

from torch import nn

from src.common.cnn_image_classifier import CnnImageClassifier
from src.common.model_creator import ModelCreator
from src.model_creation.densenet121_cnn_creator import Densenet121CnnCreator


class Densenet121CnnClassifier(CnnImageClassifier):
    @property
    def name(self) -> str:
        return 'densenet18cnn'

    def _get_loss_criterion(self):
        return nn.CrossEntropyLoss()

    @property
    def model_creator_class(self) -> Type[ModelCreator]:
        return Densenet121CnnCreator

    def unfreeze_classifier_layer(self):
        self.unfreeze_layer('classifier')

    def freeze_all_layers_except_fc(self):
        self.freeze_all_layers()
        self.unfreeze_classifier_layer()
