import torch
from torch import nn, Tensor, distributions

from nflows.transforms import Transform, IdentityTransform

from models.autoencoder import GaussianEncoder, GaussianDecoder


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

    def encode(self, x: Tensor):
        raise NotImplementedError

    def decode(self, z: Tensor):
        raise NotImplementedError

    def sample(self, n_samples: int):
        raise NotImplementedError

    def forward(self, x: Tensor):
        raise NotImplementedError

    def loss_function(self, x: Tensor):
        raise NotImplementedError


class GaussianAutoEncoder(AutoEncoder):
    def __init__(self, encoder: GaussianEncoder, decoder: GaussianDecoder):
        super(GaussianAutoEncoder, self).__init__()

        assert encoder.latent_dim == decoder.latent_dim

        self.encoder = encoder
        self.decoder = decoder
        self.latent_dims = encoder.latent_dim
        self.prior = distributions.normal.Normal(torch.zeros(self.latent_dims),
                                                 torch.ones(self.latent_dims))
        self.eps = 1e-6
        self.device = None

    def encode(self, x: Tensor):
        return self.encoder(x)

    def decode(self, z: Tensor):
        return self.decoder(z)

    def sample(self, num_samples: int = 1, temperature: int = 1, z: Tensor = None):
        self.set_device()
        if z is None:
            z = self.prior.sample((num_samples, )) * temperature
        return self.decode(z)[0]

    def set_device(self):
        # TODO: override to? see https://stackoverflow.com/questions/59179609/how-to-make-a-pytorch-distribution-on-gpu
        if self.device is None:
            self.device = next(self.encoder.parameters()).device
            # Putting loc and scale to device gives nans for some reason
            self.prior = distributions.normal.Normal(torch.zeros(self.latent_dims).to(self.device),
                                                     torch.ones(self.latent_dims).to(self.device))

class ExtendedGaussianAutoEncoder(GaussianAutoEncoder):
    def __init__(self, encoder: GaussianEncoder, decoder: GaussianDecoder, 
                 posterior_bijector: Transform = IdentityTransform(), prior_bijector: Transform = IdentityTransform()):
        super(ExtendedGaussianAutoEncoder, self).__init__(encoder, decoder)
        self.posterior_bijector = posterior_bijector
        self.prior_bijector = prior_bijector
        