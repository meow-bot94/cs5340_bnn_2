from typing import Type

import pyro

from src.common.model_creator import ModelCreator
from src.common.pyro_image_classifier import PyroImageClassifier
from src.model_creation.pyro_densenet121_bnn_creator import \
    PyroDensenet121BnnCreator


class PyroDensenet121BnnClassifier(PyroImageClassifier):

    @property
    def name(self) -> str:
        return 'pyro_densenet121bnn'

    @property
    def model_creator_class(self) -> Type[ModelCreator]:
        return PyroDensenet121BnnCreator

    def _get_loss_criterion(self):
        return pyro.infer.Trace_ELBO(
            # num_particles=self.num_samples,
        )

    @property
    def num_samples(self) -> int:
        return 10

    def unfreeze_layer(self, layer_name: str):
        target_layer = self._model._net
        for layer_section in layer_name.split('.'):
            target_layer = getattr(target_layer, layer_section)
        for param in target_layer.parameters():
            param.requires_grad = True
