from typing import Type

import pyro

from src.common.model_creator import ModelCreator
from src.common.pyro_image_classifier import PyroImageClassifier
from src.model_creation.pyro_resnet18_bnn_creator import PyroResnet18BnnCreator


class PyroResnet18BnnClassifier(PyroImageClassifier):

    @property
    def name(self) -> str:
        return 'pyro_resnet18bnn'

    @property
    def model_creator_class(self) -> Type[ModelCreator]:
        return PyroResnet18BnnCreator

    def _get_loss_criterion(self):
        return pyro.infer.Trace_ELBO(
            # num_particles=self.num_samples,
        )

    @property
    def num_samples(self) -> int:
        return 10

    def unfreeze_layer(self, layer_name: str):
        for param in getattr(self._model._resnet, layer_name).parameters():
            param.requires_grad = True

    def unfreeze_fc_layer(self):
        self.unfreeze_layer('fc')
        self.unfreeze_layer('logsoftmax')

    def freeze_all_layers_except_fc(self):
        self.freeze_all_layers()
        self.unfreeze_fc_layer()
