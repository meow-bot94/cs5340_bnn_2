import pyro
from pyro.distributions import Categorical


class PyroLikelihoodGetter:
    def __init__(self, model, data_size: int):
        self._model = model
        self._data_size = data_size

    def _get_likelihood(self, model, data_size: int):
        def pyro_likelihood(x, y=None):
            logits = model(x)
            with pyro.plate("data_plate", x.shape[0]):
                pyro.sample(
                    "data",
                    Categorical(logits=logits).to_event(),
                    obs=y
                )
            return logits

        return pyro_likelihood

    def get(self):
        return self._get_likelihood(self._model, self._data_size)
