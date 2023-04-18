from torch.utils.data import DataLoader

from src.common.image_classifier import ImageClassifier
from src.prediction.cnn_batch_predictor import CnnBatchPredictor
from src.training.cnn_trainer import CnnTrainer


class CnnImageClassifier(ImageClassifier):
    def _fit(self, num_epoch: int, optimizer, verbose):
        assert self._model is not None, 'Model not initiated. Run init first.'
        trainer = CnnTrainer(
            self._model,
            self._dataset,
            self._get_loss_criterion(),
            self._device,
            optimizer=optimizer,
        )
        return trainer.train(num_epoch, verbose)

    def predict(self, dataloader: DataLoader):
        return CnnBatchPredictor(self._model, dataloader, self._device).predict()

    def _load_model(self, model_state_dict: dict):
        self._model.load_state_dict(model_state_dict)

    @property
    def net(self):
        return self._model
