import pyro


class PyroDummyEvaluationInitializer:
    def __init__(self, model, dataloader, device):
        self._model = model
        self._dataloader = dataloader
        self._device = device

    def _get_dummy_optimizer(self):
        return pyro.optim.Adam({"lr": 1e-3})

    def _get_dummy_loss(self):
        return pyro.infer.Trace_ELBO()

    def _get_dummy_data(self):
        X_batch, y_batch = next(iter(self._dataloader))
        X_batch, y_batch = X_batch.to(self._device), y_batch.to(self._device)
        return X_batch, y_batch

    def init(self):
        optim = self._get_dummy_optimizer()
        svi = pyro.infer.SVI(
            self._model.model,
            self._model.guide,
            optim,
            self._get_dummy_loss()
        )
        X_batch, y_batch = self._get_dummy_data()
        loss = svi.evaluate_loss(X_batch, y_batch)
        return loss
