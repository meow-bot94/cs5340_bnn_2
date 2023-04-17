import pyro
import pyro.distributions
import pyro.nn
import torch
import torchvision
from pyro.distributions import Categorical
from torch import nn

from src.data_loader.dataset import Dataset


class PyroResnet18Bnn(pyro.nn.PyroModule):

    def __init__(self, dataset: Dataset, device: str):
        super().__init__()
        self._resnet = self._create_pyro_resnet(dataset, device)
        self.guide = self._get_guide()

    def forward(self, x, y=None):
        logits = self._resnet(x)
        with pyro.plate("data_plate", x.shape[0]):
            pyro.sample("data", Categorical(logits=logits).to_event(), obs=y)
        return logits

    def model(self, x, y=None):
        logits = self._resnet(x)
        with pyro.plate("data_plate", x.shape[0]):
            pyro.sample("data", Categorical(logits=logits).to_event(), obs=y)
        return logits

    def _get_guide(self):
        values = self._resnet.state_dict()
        guide = pyro.infer.autoguide.AutoNormal(
            self.model, init_scale=1e-4,
            init_loc_fn=pyro.infer.autoguide.init_to_value(values=values)
        )
        return guide

    def _create_resnet(self, dataset: Dataset, device: str):
        resnet = torchvision.models.resnet18(
            weights=torchvision.models.ResNet18_Weights.DEFAULT
        )
        for param in resnet.parameters():
            param.requires_grad = False
        resnet.class_to_idx = dataset.class_to_idx
        resnet.idx_to_class = dataset.idx_to_class
        resnet.fc = nn.Sequential(
            nn.Linear(
                resnet.fc.in_features,
                dataset.num_classes,
                bias=True,
            )
        )
        resnet.add_module('logsoftmax', nn.LogSoftmax(dim=1))
        resnet.fc.requires_grad = True
        resnet.logsoftmax.requires_grad = True
        resnet.to(device)
        return resnet

    def _convert_torch_to_pyro(self, model, device: str):
        pyro.nn.module.to_pyro_module_(model)

        # prior definition
        for module_name, m in model.named_modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                m.weight = pyro.nn.PyroSample(pyro.distributions.Normal(
                    m.weight,
                    torch.full_like(m.weight, 1e-3, device=device)
                ).to_event())
                if m.bias is not None:
                    m.bias = pyro.nn.PyroSample(pyro.distributions.Normal(
                        m.bias,
                        torch.full_like(m.bias, 1e-3, device=device)
                    ).to_event())

    def _create_pyro_resnet(self, dataset: Dataset, device: str):
        resnet = self._create_resnet(dataset, device)
        self._convert_torch_to_pyro(resnet, device)
        return resnet
