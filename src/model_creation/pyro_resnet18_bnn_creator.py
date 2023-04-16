from src.common.model_creator import ModelCreator
from src.data_loader.dataset import Dataset
from src.model_creation.pyro_resnet18_bnn import PyroResnet18Bnn


class PyroResnet18BnnCreator(ModelCreator):

    def _create(self, dataset: Dataset, device: str):
        pyro_resnet = PyroResnet18Bnn(dataset, device)
        return pyro_resnet
