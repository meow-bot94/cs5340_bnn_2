import tyxe
from tqdm import tqdm, trange
from tyxe.bnn import _to
from pyro.infer import JitTraceMeanField_ELBO, JitTrace_ELBO, SVI, \
    TraceMeanField_ELBO, Trace_ELBO


class JitVariationalBNN(tyxe.VariationalBNN):
    def fit(self, data_loader, optim, num_epochs, callback=None, num_particles=1, closed_form_kl=True, device=None):
        """Optimizes the variational parameters on data from data_loader using optim for num_epochs.

        :param Iterable data_loader: iterable over batches of data, e.g. a torch.utils.data.DataLoader. Assumes that
            each element consists of a length two tuple of list, with the first element either containing a single
            object or a list of objects, e.g. torch.Tensors, that are the inputs to the neural network. The second
            element is a single torch.Tensor e.g. of class labels.
        :param optim: pyro optimizer to be used for constructing an SVI object, e.g. pyro.optim.Adam({"lr": 1e-3}).
        :param int num_epochs: number of passes over data_loader.
        :param callable callback: optional function to invoke after every training epoch. Receives the BNN object,
            the epoch number and the average value of the ELBO over the epoch. May return True to terminate
            optimization before num_epochs, e.g. if it finds that a validation log likelihood saturates.
        :param int num_particles: number of MC samples for estimating the ELBO.
        :param bool closed_form_kl: whether to use TraceMeanField_ELBO or Trace_ELBO, i.e. calculate KL divergence
            between approximate posterior and prior in closed form or via a Monte Carlo estimate.
        :param torch.device device: optional device to send the data to.
        """
        old_training_state = self.net.training
        self.net.train(True)

        loss = TraceMeanField_ELBO(num_particles) if closed_form_kl else Trace_ELBO(num_particles)
        svi = SVI(self.model, self.guide, optim, loss=loss)

        tracking = trange(num_epochs)
        for epoch_idx in tracking:
            elbo = 0.
            avg_loss_for_epoch = elbo
            num_batch = 1
            batch_loss = None
            for num_batch, (input_data, observation_data) in enumerate(iter(data_loader), 1):
                batch_loss = svi.step(tuple(_to(input_data, device)), tuple(_to(observation_data, device))[0])
                elbo += batch_loss
                tracking.set_postfix(
                    avg_loss_for_epoch=f'{avg_loss_for_epoch:3f}',
                    batch_loss=f'{batch_loss:3f}',
                )

            avg_loss_for_epoch = elbo / num_batch
            tracking.set_postfix(
                avg_loss_for_epoch=f'{avg_loss_for_epoch:3f}',
                batch_loss=f'{batch_loss:3f}',
            )
            print(f'{avg_loss_for_epoch=}')
            # the callback can stop training by returning True
            if callback is not None and callback(self, epoch_idx, avg_loss_for_epoch):
                break

        self.net.train(old_training_state)
        return svi

    def tyxe_fit(self, data_loader, optim, num_epochs, callback=None, num_particles=1, closed_form_kl=True, device=None):
        """Optimizes the variational parameters on data from data_loader using optim for num_epochs.

        :param Iterable data_loader: iterable over batches of data, e.g. a torch.utils.data.DataLoader. Assumes that
            each element consists of a length two tuple of list, with the first element either containing a single
            object or a list of objects, e.g. torch.Tensors, that are the inputs to the neural network. The second
            element is a single torch.Tensor e.g. of class labels.
        :param optim: pyro optimizer to be used for constructing an SVI object, e.g. pyro.optim.Adam({"lr": 1e-3}).
        :param int num_epochs: number of passes over data_loader.
        :param callable callback: optional function to invoke after every training epoch. Receives the BNN object,
            the epoch number and the average value of the ELBO over the epoch. May return True to terminate
            optimization before num_epochs, e.g. if it finds that a validation log likelihood saturates.
        :param int num_particles: number of MC samples for estimating the ELBO.
        :param bool closed_form_kl: whether to use TraceMeanField_ELBO or Trace_ELBO, i.e. calculate KL divergence
            between approximate posterior and prior in closed form or via a Monte Carlo estimate.
        :param torch.device device: optional device to send the data to.
        """
        old_training_state = self.net.training
        self.net.train(True)

        loss = JitTraceMeanField_ELBO(num_particles) if closed_form_kl else JitTrace_ELBO(num_particles)
        svi = SVI(self.model, self.guide, optim, loss=loss)

        for i in range(num_epochs):
            print(f'epoch: {i}')
            elbo = 0.
            num_batch = 1
            for num_batch, (input_data, observation_data) in enumerate(iter(data_loader), 1):
                elbo += svi.step(tuple(_to(input_data, device)), tuple(_to(observation_data, device))[0])

            # the callback can stop training by returning True
            if callback is not None and callback(self, i, elbo / num_batch):
                break

        self.net.train(old_training_state)
        return svi