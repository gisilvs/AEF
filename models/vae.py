from typing import List

import torch
from torch import Tensor, distributions

from models.autoencoder_base import AutoEncoder
from models.autoencoder import ConvolutionalEncoder, IndependentVarianceDecoder

class VAE(AutoEncoder):
    def __init__(self, hidden_channels: int, latent_dim: int, input_dim: List):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.eps = 1e-5

        self.encoder = ConvolutionalEncoder(hidden_channels, latent_dim, input_dim)
        self.decoder = IndependentVarianceDecoder(hidden_channels, latent_dim, input_dim)

        self.prior = distributions.normal.Normal(torch.zeros(latent_dim), torch.ones(latent_dim))
        self.device = None

    def encode(self, x: Tensor):
        return self.encoder(x) # Encoder returns mu and log(sigma)

    def decode(self, z: Tensor):
        decoded_mu, decoded_sigma = self.decoder(z)
        return decoded_mu, decoded_sigma

    def sample(self, num_samples: int):
        z = self.prior.sample((num_samples, )).to(self.get_device())
        return self.decode(z)[0]

    def forward(self, x: Tensor):
        z_mu, z_sigma = self.encode(x)
        z = distributions.normal.Normal(z_mu, z_sigma).rsample().to(self.get_device())
        x_mu, x_sigma = self.decode(z)
        return x_mu, x_sigma, z_mu, z_sigma

    def loss_function(self, x: Tensor):
        x_mu, x_sigma, z_mu, z_sigma = self.forward(x)
        reconstruction_loss = -torch.distributions.normal.Normal(x_mu,x_sigma+self.eps).log_prob(x).sum([1,2,3])
        q_z = distributions.normal.Normal(z_mu, z_sigma)
        p_z = distributions.normal.Normal(torch.zeros(self.latent_dim).to(self.get_device()),
                                          torch.ones(self.latent_dim).to(self.get_device()))

        kl_div = distributions.kl.kl_divergence(q_z,p_z).sum(1)
        return reconstruction_loss + kl_div

    def get_device(self):
        if self.device is None:
            self.device = next(self.encoder.parameters()).device
        return self.device
