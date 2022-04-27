import torch
from torch import Tensor, distributions

from bijectors.masked_autoregressive_transform import get_masked_autoregressive_transform
from models.autoencoder import GaussianEncoder, GaussianDecoder
from models.autoencoder_base import GaussianAutoEncoder


class VAEIAF(GaussianAutoEncoder):
    def __init__(self, encoder: GaussianEncoder, decoder: GaussianDecoder):
        super().__init__(encoder, decoder)

        self.latent_dim = encoder.latent_dim
        self.eps = 1e-5
        self.prior = distributions.normal.Normal(torch.zeros(self.latent_dim), torch.ones(self.latent_dim))
        self.device = None
        self.iaf = get_masked_autoregressive_transform(self.latent_dim, 256, 8, 2, act_norm_between_layers=True, is_inverse=True)

    def encode(self, x: Tensor):
        return self.encoder(x)  # Encoder returns mu and log(sigma)

    def decode(self, z: Tensor):
        decoded_mu, decoded_sigma = self.decoder(z)
        return decoded_mu, decoded_sigma

    def sample(self, num_samples: int = 1, temperature: float = 1, z: Tensor = None):
        # multiplying by the temperature works like the reparametrization trick,
        # only if the prior is N(0,1)
        if z is None:
            z = self.prior.sample((num_samples,)).to(self.get_device()) * temperature

        return self.decode(z)[0]

    def forward(self, x: Tensor):
        z_mu, z_sigma = self.encode(x)
        z0 = distributions.normal.Normal(z_mu, z_sigma + self.eps).rsample().to(self.get_device())
        z, log_j_z = self.iaf.forward(z0)
        x_mu, x_sigma = self.decode(z)
        return x_mu, x_sigma, z_mu, z_sigma, z0, log_j_z, z

    def loss_function(self, x: Tensor):
        x_mu, x_sigma, z_mu, z_sigma, z0, log_j_z, z = self.forward(x)
        reconstruction_loss = torch.distributions.normal.Normal(x_mu, x_sigma + self.eps).log_prob(x).sum([1, 2, 3])
        q_z = distributions.normal.Normal(z_mu, z_sigma + self.eps).log_prob(z0).sum(-1) - log_j_z
        p_z = distributions.normal.Normal(0, 1).log_prob(z).sum(-1)
        return -(reconstruction_loss + p_z - q_z)

    def get_device(self):
        if self.device is None:
            self.device = next(self.encoder.parameters()).device
        return self.device
