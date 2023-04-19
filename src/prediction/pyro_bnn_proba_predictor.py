import torch
from pyro.infer import Predictive

from src.common.model_predictor import ModelPredictor


class PyroBnnProbaPredictor(ModelPredictor):
    def __init__(
            self,
            model,
            device,
    ):
        self._model = model
        self._device = device

    def _get_proba(
            self,
            model, x, likelihood, guide, num_samples, device,
    ) -> torch.tensor:
        if x.dim() == 3:
            x_batch = torch.tensor([x]).to(device)
        elif x.dim() == 4:
            x_batch = x.to(device)
        else:
            raise NotImplementedError(f'x dimension is not allowed: {x.dim()=}')

        model.eval()
        predictive = Predictive(
            likelihood,
            guide=guide,
            num_samples=num_samples,
        )
        samples = predictive(x_batch)

        return samples

    def get_proba(self, x: torch.tensor, num_samples: int) -> torch.tensor:
        return self._get_proba(
            self._model,
            x,
            self._model.model,
            self._model.guide,
            num_samples,
            self._device,
        )