from torch import nn
from torchvision import models

from src.common.model_creator import ModelCreator
from src.data_loader.dataset import Dataset


class Resnet50CnnCreator(ModelCreator):

    def _create(self, dataset: Dataset, device: str):
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        for param in resnet.parameters():
            param.requires_grad = False

        resnet.fc = nn.Linear(resnet.fc.in_features, len(dataset.class_to_idx))
        resnet.class_to_idx = dataset.class_to_idx
        resnet.idx_to_class = dataset.idx_to_class
        resnet.to(device)
        return resnet
