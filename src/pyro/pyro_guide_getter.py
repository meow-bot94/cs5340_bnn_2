import pyro
from pyro.infer.autoguide import AutoNormal


class PyroGuideGetter:
    def __init__(self, model, init_scale: float = 1e-4):
        self._model = model
        self._init_scale = init_scale

    def _get_guide(self, model, init_scale: float = 1e-4):
        values = model.state_dict()
        guide = AutoNormal(
            model, init_scale=init_scale,
            init_loc_fn=pyro.infer.autoguide.init_to_value(values=values))
        return guide

    def get(self):
        return self._get_guide(self._model, self._init_scale)
