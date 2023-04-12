from abc import ABC, abstractmethod


class ImageClassifier(ABC):
    def __init__(self, device: str):
        self._device = device

    @abstractmethod
    def _init(self):
        pass

    def init(self):
        return self

    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def predict(self):
        pass

