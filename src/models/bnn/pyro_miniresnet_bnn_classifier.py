from typing import Type

import pyro

from src.common.model_creator import ModelCreator
from src.common.pyro_image_classifier import PyroImageClassifier
from src.model_creation.pyro_miniresnet_bnn_creator import \
    PyroMiniresnetBnnCreator


class PyroMiniresnetBnnClassifier(PyroImageClassifier):

    @property
    def name(self) -> str:
        return 'pyro_miniresnet'

    @property
    def model_creator_class(self) -> Type[ModelCreator]:
        return PyroMiniresnetBnnCreator

    def _get_loss_criterion(self):
        return pyro.infer.Trace_ELBO(
            # num_particles=self.num_samples,
        )

    @property
    def num_samples(self) -> int:
        return 10

    @property
    def net(self):
        return self._model._resnet

    def unfreeze_fc_layer(self):
        self.unfreeze_layer('fc')
        self.unfreeze_layer('logsoftmax')

    def freeze_all_layers_except_fc(self):
        self.freeze_all_layers()
        self.unfreeze_fc_layer()
