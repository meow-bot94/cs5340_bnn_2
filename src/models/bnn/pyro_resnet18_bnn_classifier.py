from typing import Dict, Type

import pyro
from torch.utils.data import DataLoader

from src.common.image_classifier import ImageClassifier
from src.common.model_creator import ModelCreator
from src.model_creation.pyro_resnet18_bnn_creator import PyroResnet18BnnCreator
from src.prediction.pyro_bnn_batch_predictor import PyroBnnBatchPredictor
from src.pyro.pyro_dummy_evaluation_initializer import \
    PyroDummyEvaluationInitializer
from src.scoring.class_scorer import ClassScorer
from src.training.pyro_bnn_trainer import PyroBnnTrainer


class PyroResnet18BnnClassifier(ImageClassifier):

    @property
    def name(self) -> str:
        return 'pyro_resnet18bnn'

    @property
    def model_creator_class(self) -> Type[ModelCreator]:
        return PyroResnet18BnnCreator

    def _get_loss_criterion(self):
        return pyro.infer.Trace_ELBO()

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

    @property
    def num_samples(self) -> int:
        return 10

    def predict(self, dataloader: DataLoader):
        return PyroBnnBatchPredictor(
            self._model,
            dataloader,
            self.num_samples,
            self._device,
        ).predict()

    def score(self, dataloader: DataLoader):
        y_true, y_pred = self.predict(dataloader)
        return ClassScorer.score(y_true, y_pred)

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
