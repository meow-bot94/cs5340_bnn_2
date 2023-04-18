from typing import List, Tuple

import torch
from pyro.infer import Predictive

from src.common.model_predictor import ModelPredictor


class PyroBnnBatchPredictor(ModelPredictor):
    def __init__(
            self,
            model,
            dataloader,
            num_samples: int,
            device,
    ):
        self._model = model
        self._dataloader = dataloader
        self._num_samples = num_samples
        self._device = device

    def _get_prediction(self, x, likelihood, guide, num_samples):
        predictive = Predictive(
            likelihood,
            guide=guide,
            num_samples=num_samples,
            return_sites=("obs", "_RETURN")
        )  # "linear.weight"
        samples = predictive(x)
        predicted_classes = samples['_RETURN'].cpu().numpy()
        return predicted_classes

    def _predict(
            self, model, dataloader, likelihood, guide, num_samples, device,
    ) -> Tuple[List[float], List[float]]:
        y_true, y_pred = [], []

        model.eval()
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            prediction = self._get_prediction(
                X_batch, likelihood, guide, num_samples)
            best_prediction = prediction.argmax(axis=2)
            actual_label = y_batch.expand(
                num_samples, y_batch.shape[0]
            ).cpu().numpy()

            y_true += list(actual_label.flatten().tolist())
            y_pred += list(best_prediction.flatten().tolist())

        return y_true, y_pred

    def _get_probabilities(
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
        prediction = self._get_prediction(
            x_batch, likelihood, guide, num_samples)

        return prediction

    def get_proba(self, x: torch.tensor) -> torch.tensor:
        return self._get_probabilities(
            self._model,
            x,
            self._model.model,
            self._model.guide,
            self._num_samples,
            self._device,
        )

    def predict(self):
        return self._predict(
            self._model,
            self._dataloader,
            self._model.model,
            self._model.guide,
            self._num_samples,
            self._device,
        )
