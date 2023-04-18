from src.common.model_creator import ModelCreator
from src.data_loader.dataset import Dataset
from src.model_creation.pyro_densenet121_bnn import PyroDensenet121Bnn


class PyroDensenet121BnnCreator(ModelCreator):

    def _create(self, dataset: Dataset, device: str):
        pyro_bnn = PyroDensenet121Bnn(dataset, device)
        return pyro_bnn
