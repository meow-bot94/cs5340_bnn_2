from typing import List, Tuple

import torch
from torch import nn

from src.common.model_predictor import ModelPredictor


class CnnBatchPredictor(ModelPredictor):
    def __init__(self, model, dataloader, device):
        self._model = model
        self._dataloader = dataloader
        self._device = device

    def _predict(
            self, model, dataloader, device,
    ) -> Tuple[List[float], List[float]]:
        y_true, y_pred = [], []

        model.eval()
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            logits = model(X_batch)
            log_probs = nn.functional.log_softmax(logits, dim=0)
            y_batch_pred = torch.argmax(log_probs, dim=1)
            y_true += list(y_batch.cpu())
            y_pred += list(y_batch_pred.cpu())

        return y_true, y_pred

    def predict(self):
        return self._predict(self._model, self._dataloader, self._device)
