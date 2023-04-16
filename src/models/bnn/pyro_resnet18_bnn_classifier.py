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
        return pyro.infer.Trace_ELBO()

    @property
    def num_samples(self) -> int:
        return 10
