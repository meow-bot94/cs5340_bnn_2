from torch import nn
from torchvision import models

from src.common.model_creator import ModelCreator
from src.data_loader.dataset import Dataset


class Densenet121CnnCreator(ModelCreator):

    def _create(self, dataset: Dataset, device: str):
        densenet = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        for param in densenet.parameters():
            param.requires_grad = False

        densenet.classifier = nn.Linear(
            densenet.classifier.in_features,
            len(dataset.class_to_idx),
        )
        densenet.class_to_idx = dataset.class_to_idx
        densenet.idx_to_class = dataset.idx_to_class
        densenet.to(device)
        return densenet
