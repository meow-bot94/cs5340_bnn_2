from abc import abstractmethod
from typing import Dict

import pyro
from torch.utils.data import DataLoader

from src.common.image_classifier import ImageClassifier
from src.prediction.pyro_bnn_batch_predictor import PyroBnnBatchPredictor
from src.pyro.pyro_dummy_evaluation_initializer import \
    PyroDummyEvaluationInitializer
from src.training.pyro_bnn_trainer import PyroBnnTrainer


class PyroImageClassifier(ImageClassifier):
    @property
    @abstractmethod
    def num_samples(self) -> int:
        return 1

    def _fit(self, num_epoch: int, verbose):
        assert self._model is not None, 'Model not initiated. Run init first.'
        pyro.clear_param_store()
        trainer = PyroBnnTrainer(
            self._model,
            self._dataset,
            self.num_samples,
            self._get_loss_criterion(),
            self._device,
        )
        return trainer.train(num_epoch, verbose)

    def predict(self, dataloader: DataLoader):
        return PyroBnnBatchPredictor(
            self._model,
            dataloader,
            self.num_samples,
            self._device,
        ).predict()

    def _init_pyro_with_dummy_eval(self, model, dataloader, device):
        return PyroDummyEvaluationInitializer(
            model,
            dataloader,
            device,
        ).init()

    @property
    def pyro_state_dict_key(self) -> str:
        return 'pyro_state_dict'

    def save_model(self, result_dict: Dict):
        result_dict[self.pyro_state_dict_key] = pyro.get_param_store()
        super().save_model(result_dict)

    def _load_model(self, model_state_dict: dict, pyro_state_dict):
        pyro.clear_param_store()
        self._init_pyro_with_dummy_eval(
            self._model,
            self._dataset.train_dataloader,
            self._device,
        )
        self._model.load_state_dict(model_state_dict)
        pyro_state_dict.save('tmp.save')
        pyro.get_param_store().load('tmp.save')
