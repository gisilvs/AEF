from typing import List

import torch
from torch import Tensor, distributions

from models.autoencoder_base import GaussianAutoEncoder
from models.autoencoder import GaussianEncoder, GaussianDecoder


class VAE(GaussianAutoEncoder):
    def __init__(self, encoder: GaussianEncoder, decoder: GaussianDecoder):
        super().__init__(encoder, decoder)

        self.latent_dim = encoder.latent_dim
        self.eps = 1e-5
        self.prior = distributions.normal.Normal(torch.zeros(self.latent_dim), torch.ones(self.latent_dim))
        self.device = None

    def encode(self, x: Tensor):
        return self.encoder(x) # Encoder returns mu and log(sigma)

    def decode(self, z: Tensor):
        decoded_mu, decoded_sigma = self.decoder(z)
        return decoded_mu, decoded_sigma

    def sample(self, num_samples: int, temperature=1):
        # multiplying by the temperature works like the reparametrization trick,
        # only if the prior is N(0,1)
        z = self.prior.sample((num_samples, )).to(self.get_device()) * temperature
        return self.decode(z)[0]

    def forward(self, x: Tensor):
        z_mu, z_sigma = self.encode(x)
        z = distributions.normal.Normal(z_mu, z_sigma+self.eps).rsample().to(self.get_device())
        x_mu, x_sigma = self.decode(z)
        return x_mu, x_sigma, z_mu, z_sigma

    def loss_function(self, x: Tensor):
        x_mu, x_sigma, z_mu, z_sigma = self.forward(x)
        reconstruction_loss = torch.distributions.normal.Normal(x_mu, x_sigma+self.eps).log_prob(x).sum([1,2,3])
        q_z = distributions.normal.Normal(z_mu, z_sigma+self.eps)

        kl_div = distributions.kl.kl_divergence(q_z, self.prior).sum(1)
        return -(reconstruction_loss - kl_div)

    def get_device(self):
        if self.device is None:
            self.device = next(self.encoder.parameters()).device
            self.prior = distributions.normal.Normal(torch.zeros(self.latent_dim).to(self.device),
                                                     torch.ones(self.latent_dim).to(self.device))
        return self.device
