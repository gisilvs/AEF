from torch import nn, Tensor

from models.autoencoder import GaussianDecoder, GaussianEncoder


class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()

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
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

        assert encoder.latent_dim == decoder.latent_dim
