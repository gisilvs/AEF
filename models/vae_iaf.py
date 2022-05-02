import torch
from torch import Tensor, distributions

from bijectors.masked_autoregressive_transform import get_masked_autoregressive_transform
from models.autoencoder import GaussianEncoder, GaussianDecoder
from models.autoencoder_base import GaussianAutoEncoder
from models.vae import ExtendedVAE


class VAEIAF(GaussianAutoEncoder):
    def __init__(self, encoder: GaussianEncoder, decoder: GaussianDecoder):
        super(VAEIAF, self).__init__(encoder, decoder)
        self.iaf = get_masked_autoregressive_transform(self.latent_dim, 256, 8, 2, act_norm_between_layers=True, is_inverse=True)

    def forward(self, x: Tensor):
        self.set_device()
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
        p_z = distributions.normal.Normal(0, 1).log_prob(z).sum(-1)
        return -(reconstruction_loss + p_z - q_z)


class ExtendedVAEIAF(ExtendedVAE):
    def __init__(self, encoder: GaussianEncoder, decoder: GaussianDecoder, hidden_features: int = 256,
                 num_layers: int = 8, num_blocks_per_layer: int = 2):
        iaf = get_masked_autoregressive_transform(features=encoder.latent_dim,
                                                  hidden_features=hidden_features, num_layers=num_layers,
                                                  num_blocks_per_layer=num_blocks_per_layer,
                                                  act_norm_between_layers=True,
                                                  is_inverse=True)
        super(VAEIAF, self).__init__(encoder, decoder, posterior_bijector=iaf)


class DenoisingVAEIAF(VAEIAF):
    def loss_function(self, x_noisy: Tensor, x_original: Tensor):
        x_mu, x_sigma, z_mu, z_sigma, z0, log_j_z, z = self.forward(x_noisy)
        reconstruction_loss = torch.distributions.normal.Normal(x_mu, x_sigma + self.eps).log_prob(x_original).sum([1, 2, 3])
        q_z = distributions.normal.Normal(z_mu, z_sigma + self.eps).log_prob(z0).sum(-1) - log_j_z
        p_z = distributions.normal.Normal(0, 1).log_prob(z).sum(-1)
        return -(reconstruction_loss + p_z - q_z)
