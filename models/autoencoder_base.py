from typing import List

import torch
from torch import nn, Tensor, distributions
from nflows.transforms import Transform, IdentityTransform

from models.autoencoder import GaussianEncoder, GaussianDecoder, Coder

from torch.distributions.normal import Normal


class AutoEncoder(nn.Module):
    """ Base class for any autoencoder. """
    def __init__(self, encoder: Coder, decoder: Coder):
        super(AutoEncoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def encode(self, x: Tensor):
        """
        Encode input from data space into latent space.
        """
        raise NotImplementedError

    def decode(self, z: Tensor):
        """
        Decode input from latent space back to data space.
        """
        raise NotImplementedError

    def forward(self, x: Tensor):
        """
        Reconstruct input by first encoding and then decoding it.
        """
        raise NotImplementedError

    def loss_function(self, x: Tensor):
        raise NotImplementedError


class GaussianAutoEncoder(AutoEncoder):
    def __init__(self, encoder: GaussianEncoder, decoder: GaussianDecoder):
        super(GaussianAutoEncoder, self).__init__(encoder, decoder)

        assert encoder.latent_dims == decoder.latent_dims

        self.encoder = encoder
        self.decoder = decoder
        self.latent_dims = encoder.latent_dims
        self.prior = Normal(torch.zeros(self.latent_dims), torch.ones(self.latent_dims))
        self.eps = 1e-6

    def encode(self, x: Tensor):
        return self.encoder(x)

    def decode(self, z: Tensor):
        return self.decoder(z)

    def sample(self, num_samples: int = 1, temperature: int = 1):
        """
        Function to sample from prior distribution and decode it to return a sample (ideally) from the data distribution.
        """
        z = self.prior.sample((num_samples,)) * temperature
        x, _ = self.decoder(z)
        return x

    def to(self, *args, **kwargs):
        """
        Since the prior distribution is not sent to device when the model is sent to device, we override to().
        """
        super().to(*args, **kwargs)
        self.prior = Normal(torch.zeros(self.latent_dims).to(*args, **kwargs),
                            torch.ones(self.latent_dims).to(*args, **kwargs))
        return self


class ExtendedGaussianAutoEncoder(GaussianAutoEncoder):
    def __init__(self, encoder: GaussianEncoder, decoder: GaussianDecoder,
                 posterior_bijector: Transform = IdentityTransform(), prior_bijector: Transform = IdentityTransform(),
                 preprocessing_layers: List = ()):
        super(ExtendedGaussianAutoEncoder, self).__init__(encoder, decoder)
        self.posterior_bijector = posterior_bijector
        self.prior_bijector = prior_bijector
        self.preprocessing_layers = nn.ModuleList(preprocessing_layers)

    def encode(self, x: Tensor):
        for layer in self.preprocessing_layers:
            x, _ = layer.inverse(x)
        z0, _ = self.encoder(x)
        z, _ = self.posterior_bijector.forward(z0)
        return z

    def decode(self, z: Tensor, from_base: bool = False):
        """
        :param z: sample from the latent space
        :param from_base: boolean value to determine whether we need to put z through the prior bijector in the
        forward direction. In other words, if from_base is True then we assume z is from the base distribution of the
        prior flow, if from_base is False then we assume z is from the latent space and we only decode it.
        :return:
        """
        if from_base:
            z, _ = self.prior_bijector.forward(z)
        x, _ = self.decoder(z)
        for i in range(len(self.preprocessing_layers) - 1, -1, -1):
            x, _ = self.preprocessing_layers[i](x)
        return x

    def sample(self, num_samples: int = 1, temperature: float = 1):
        z0 = self.prior.sample((num_samples,)) * temperature
        z, _ = self.prior_bijector.forward(z0)
        x, _ = self.decoder(z)
        for i in range(len(self.preprocessing_layers) - 1, -1, -1):
            x, _ = self.preprocessing_layers[i](x)
        return x

