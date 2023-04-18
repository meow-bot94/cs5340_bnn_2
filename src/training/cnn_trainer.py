from typing import Dict, Tuple

import pandas as pd
from torch import nn, optim
from torch.utils.data import DataLoader

from src.common.model_trainer import ModelTrainer
from src.prediction.cnn_batch_predictor import CnnBatchPredictor
from src.scoring.epoch_score_printer import EpochScorePrinter


class CnnTrainer(ModelTrainer):
    def _get_optimizer(self, model):
        return optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    def get_optimizer(self):
        return self._get_optimizer(self._model)

    def _get_predictor(self, model, dataloader: DataLoader, device: str):
        return CnnBatchPredictor(
            self._model, self._dataset.test_dataloader, self._device
        )

    def _train_epoch(self, model, loader, optimizer, criterion, device: str):

        # Initialize epoch loss (cumulative loss of all batch)
        epoch_loss = 0.0

        model.train()
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            logits = model(X_batch)

            loss = criterion(logits, y_batch)
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            optimizer.zero_grad()

            # Keep track of overall epoch loss
            epoch_loss += loss.item()

        return epoch_loss
