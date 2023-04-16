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
    ):
        self._num_samples = num_samples
        super().__init__(model, dataset, loss_criterion, device)
        self._loss_function = loss_criterion
        self._loss_criterion = self._create_loss_criterion(self._model,
                                                           self._loss_function)

    def _get_optimizer(self, model):
        return pyro.optim.ClippedAdam({"lr": 1e-3})

    def get_optimizer(self):
        return self._get_optimizer(self._model)

    def _create_loss_criterion(self, model, loss_function):
        optimizer = self._get_optimizer(model)
        svi = pyro.infer.SVI(model.model, model.guide, optimizer, loss_function)
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
