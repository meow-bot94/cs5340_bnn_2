from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from src.data_loader.dataset import Dataset
from src.scoring.class_scorer import ClassScorer
from src.scoring.epoch_score_printer import EpochScorePrinter


class ModelTrainer(ABC):
    def __init__(self, model, dataset: Dataset, loss_criterion, device: str):
        self._model = model
        self._dataset = dataset
        self._loss_criterion = loss_criterion
        self._device = device
        self._predictor = self._get_predictor(
            self._model, self._dataset.test_dataloader, self._device)
        self.best_metrics = self._init_best_metrics()
        self.best_models = self._init_initial_best_models()

    @abstractmethod
    def _get_predictor(self, model, dataloader: DataLoader, device: str):
        pass

    @abstractmethod
    def _train_epoch(self, model, loader, optimizer, criterion, device: str):
        pass

    @abstractmethod
    def _get_optimizer(self, model):
        pass

    def _score(self) -> Dict[str, float]:
        y_true, y_pred = self._predictor.predict()
        return ClassScorer.score(y_true, y_pred)

    @property
    def metrics(self) -> List[str]:
        return ['loss', 'f1_score', 'accuracy_score']

    @property
    def score_metrics(self) -> List[str]:
        return ['f1_score', 'accuracy_score']

    def _init_best_metrics(self) -> Dict[str, float]:
        initial_metrics = dict(zip(self.metrics, [0 for _ in self.metrics]))
        initial_metrics['loss'] = np.inf
        return initial_metrics

    def _init_initial_best_models(self) -> Dict[str, None]:
        return dict.fromkeys(self.metrics)

    def _update_best_models(self, result: Dict[str, float], model):
        loss = result['loss']
        if loss < self.best_metrics['loss']:
            self.best_metrics['loss'] = loss
            self.best_models['loss'] = model.state_dict()
        for metric in self.score_metrics:
            if result[metric] > self.best_metrics[metric]:
                self.best_metrics[metric] = result[metric]
                self.best_models[metric] = model.state_dict()

    def _format_best_result_keys(self, best_result_dict: Dict) -> Dict:
        return {
            f'best_{metric}': model_state
            for metric, model_state in best_result_dict.items()
        }

    def _format_best_model_keys(self, best_model_dict: Dict) -> Dict:
        return {
            f'best_{metric}_model': model_state
            for metric, model_state in best_model_dict.items()
        }

    def _format_output_results(self):
        best_metrics = self._format_best_result_keys(self.best_metrics)
        best_model_dict = self._format_best_model_keys(self.best_models)
        return {
            **best_metrics,
            **best_model_dict,
        }

    def _train(
            self,
            model,
            num_epochs,
            train_dataloader,
            test_dataloader,
            optimizer, criterion,
            device,
            verbose
    ) -> Tuple[pd.DataFrame, Dict]:
        results = []

        for epoch in range(num_epochs):
            epoch_loss = self._train_epoch(
                model,
                train_dataloader,
                optimizer,
                criterion,
                device,
            )
            epoch_result = self._score()
            epoch_result['loss'] = epoch_loss
            results.append(epoch_result)

            self._update_best_models(epoch_result, model)

            EpochScorePrinter.print(
                epoch, num_epochs,
                verbose,
                **epoch_result,
            )
        results_df = pd.DataFrame.from_records(results)
        best_results = self._format_output_results()
        return results_df, best_results

    def train(
            self,
            num_epoch: int,
            verbose=True,
    ) -> Tuple[pd.DataFrame, Dict]:
        optimizer = self._get_optimizer(self._model)
        return self._train(
            self._model,
            num_epoch,
            self._dataset.train_dataloader,
            self._dataset.test_dataloader,
            optimizer,
            self._loss_criterion,
            self._device,
            verbose=verbose
        )
