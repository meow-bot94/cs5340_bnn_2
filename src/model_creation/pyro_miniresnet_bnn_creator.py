from src.common.model_creator import ModelCreator
from src.data_loader.dataset import Dataset
from src.model_creation.pyro_miniresnet_bnn import PyroMiniresnetBnn


class PyroMiniresnetBnnCreator(ModelCreator):

    def _create(self, dataset: Dataset, device: str):
        pyro_resnet = PyroMiniresnetBnn(dataset, device)
        return pyro_resnet
