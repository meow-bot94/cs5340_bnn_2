import copy
from typing import Dict

import pyro
from torch.utils.data import DataLoader

from src.common.model_trainer import ModelTrainer
from src.data_loader.dataset import Dataset
from src.prediction.pyro_bnn_batch_predictor import PyroBnnBatchPredictor


class PyroBnnTrainer(ModelTrainer):
    def __init__(
            self,
            model,
            dataset: Dataset,
            num_samples,
            loss_criterion,
            device: str,
            optimizer=None,
    ):
        self._num_samples = num_samples
        super().__init__(model, dataset, loss_criterion, device, optimizer)
        self._loss_function = loss_criterion
        self._loss_criterion = self._create_loss_criterion(
            self._model,
            self._loss_function,
        )
        self.best_model_pyro_params = self._init_initial_best_pyro_params()

    def _init_initial_best_pyro_params(self) -> Dict[str, None]:
        return dict.fromkeys(self.metrics)

    def _update_best_models(self, result: Dict[str, float], model):
        loss = result['loss']
        if loss < self.best_metrics['loss']:
            self.best_metrics['loss'] = loss
            self.best_models['loss'] = copy.deepcopy(model.state_dict())
            self.best_model_pyro_params['loss'] = copy.deepcopy(pyro.get_param_store())
        for metric in self.score_metrics:
            if result[metric] > self.best_metrics[metric]:
                self.best_metrics[metric] = result[metric]
                self.best_models[metric] = copy.deepcopy(model.state_dict())
                self.best_model_pyro_params[metric] = copy.deepcopy(pyro.get_param_store())

    def _format_best_pyro_param_keys(
            self,
            best_model_pyro_params_dict: Dict,
    ) -> Dict:
        return {
            f'best_{metric}_model_pyro_params': model_state
            for metric, model_state in best_model_pyro_params_dict.items()
        }

    def _format_output_results(self):
        best_metrics = self._format_best_result_keys(self.best_metrics)
        best_model_dict = self._format_best_model_keys(self.best_models)
        best_model_param_dict = self._format_best_pyro_param_keys(
            self.best_model_pyro_params)
        return {
            **best_metrics,
            **best_model_dict,
            **best_model_param_dict,
        }

    def _get_optimizer(self, model):
        return pyro.optim.ClippedAdam({"lr": 1e-3, })

    def get_optimizer(self):
        return self._get_optimizer(self._model)

    def _create_loss_criterion(self, model, loss_function):
        optimizer = self._get_optimizer(model)
        svi = pyro.infer.SVI(
            model.model,
            model.guide,
            optimizer,
            loss_function,
        )
        return svi

    def _get_predictor(self, model, dataloader: DataLoader, device: str):
        return PyroBnnBatchPredictor(
            self._model,
            self._dataset.test_dataloader,
            self._num_samples,
            self._device
        )

    def _train_epoch(self, model, loader, optimizer, criterion,
                     device: str):
        # Initialize epoch loss (cumulative loss of all batch)
        epoch_loss = 0.0

        model.train()
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            loss = criterion.step(X_batch, y_batch)

            # Keep track of overall epoch loss
            epoch_loss += loss

        return epoch_loss
