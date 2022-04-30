import torch
from torch import Tensor, distributions

from bijectors.masked_autoregressive_transform import get_masked_autoregressive_transform
from models.autoencoder import GaussianEncoder, GaussianDecoder
from models.autoencoder_base import GaussianAutoEncoder


class VAEIAF(GaussianAutoEncoder):
    def __init__(self, encoder: GaussianEncoder, decoder: GaussianDecoder):
        super().__init__(encoder, decoder)
        self.iaf = get_masked_autoregressive_transform(self.latent_ndims, 256, 8, 2, act_norm_between_layers=True,
                                                       is_inverse=True)

    def forward(self, x: Tensor):
        z_mu, z_sigma = self.encode(x)
        z0 = distributions.normal.Normal(z_mu, z_sigma + self.eps).rsample()
        z, log_j_z = self.iaf.forward(z0)
        x_mu, x_sigma = self.decode(z)
        return x_mu, x_sigma, z_mu, z_sigma, z0, log_j_z, z

    def loss_function(self, x: Tensor):
        self.set_device()
        x_mu, x_sigma, z_mu, z_sigma, z0, log_j_z, z = self.forward(x)
        reconstruction_loss = torch.distributions.normal.Normal(x_mu, x_sigma + self.eps).log_prob(x).sum([1, 2, 3])
        q_z = distributions.normal.Normal(z_mu, z_sigma + self.eps).log_prob(z0).sum(-1) - log_j_z
        p_z = self.prior.log_prob(z).sum(-1)
        return -(reconstruction_loss + p_z - q_z)

