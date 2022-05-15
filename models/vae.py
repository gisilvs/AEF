from typing import List

import torch
from nflows.transforms import IdentityTransform, Transform
from torch import Tensor, distributions

from models.autoencoder import GaussianEncoder, GaussianDecoder
from models.autoencoder_base import GaussianAutoEncoder, ExtendedGaussianAutoEncoder


class VAE(GaussianAutoEncoder):
    def __init__(self, encoder: GaussianEncoder, decoder: GaussianDecoder):
        super(VAE, self).__init__(encoder, decoder)

    def forward(self, x: Tensor):
        z_mu, z_sigma = self.encoder(x)
        z = distributions.normal.Normal(z_mu, z_sigma+self.eps).rsample()
        x_mu, x_sigma = self.decoder(z)
        return x_mu, x_sigma, z_mu, z_sigma

    def loss_function(self, x: Tensor):
        self.set_device()
        x_mu, x_sigma, z_mu, z_sigma = self.forward(x)
        reconstruction_loss = torch.distributions.normal.Normal(x_mu, x_sigma+self.eps).log_prob(x).sum([1,2,3])
        q_z = distributions.normal.Normal(z_mu, z_sigma+self.eps)

        kl_div = distributions.kl.kl_divergence(q_z, self.prior).sum(1)
        return -(reconstruction_loss - kl_div)


class ExtendedVAE(ExtendedGaussianAutoEncoder):
    def __init__(self, encoder: GaussianEncoder, decoder: GaussianDecoder,
                 posterior_bijector: Transform = IdentityTransform(), prior_bijector: Transform = IdentityTransform()):
        super(ExtendedVAE, self).__init__(encoder, decoder, posterior_bijector, prior_bijector)

    def forward(self, x: Tensor):
        self.set_device()
        z_mu, z_sigma = self.encoder(x)
        z0 = distributions.normal.Normal(z_mu, z_sigma + self.eps).rsample()
        z, log_j_q = self.posterior_bijector.forward(z0)
        x_mu, x_sigma = self.decoder(z)
        return x_mu, x_sigma, z_mu, z_sigma, z0, log_j_q, z

    def loss_function(self, x: Tensor):
        self.set_device()
        x_mu, x_sigma, z_mu, z_sigma, z0, log_j_q, z = self.forward(x)
        reconstruction_loss = torch.distributions.normal.Normal(x_mu, x_sigma + self.eps).log_prob(x).sum([1, 2, 3])
        q_z = distributions.normal.Normal(z_mu, z_sigma + self.eps).log_prob(z0).sum(-1) - log_j_q
        z_inv, log_j_p = self.prior_bijector.inverse(z)
        p_z = self.prior.log_prob(z_inv).sum(-1) + log_j_p
        return -(reconstruction_loss + p_z - q_z)

class DenoisingVAE(VAE):
    def __init__(self, encoder: GaussianEncoder, decoder: GaussianDecoder):
        super(DenoisingVAE, self).__init__(encoder, decoder)


    def loss_function(self, x_noisy: Tensor, x_original: Tensor):
        self.set_device()
        x_mu, x_sigma, z_mu, z_sigma = self.forward(x_noisy)
        reconstruction_loss = torch.distributions.normal.Normal(x_mu, x_sigma+self.eps).log_prob(x_original).sum([1,2,3])
        q_z = distributions.normal.Normal(z_mu, z_sigma+self.eps)

        kl_div = distributions.kl.kl_divergence(q_z, self.prior).sum(1)
        return -(reconstruction_loss - kl_div)


class ExtendedDenoisingVAE(ExtendedVAE):
    def __init__(self, encoder: GaussianEncoder, decoder: GaussianDecoder,
                 posterior_bijector: Transform = IdentityTransform(), prior_bijector: Transform = IdentityTransform()):
        super(ExtendedDenoisingVAE, self).__init__(encoder, decoder, posterior_bijector, prior_bijector)

    def loss_function(self, x_noisy: Tensor, x_original: Tensor):
        self.set_device()
        x_mu, x_sigma, z_mu, z_sigma, z0, log_j_q, z = self.forward(x_noisy)
        reconstruction_loss = torch.distributions.normal.Normal(x_mu, x_sigma + self.eps).log_prob(x_original).sum([1, 2, 3])
        q_z = distributions.normal.Normal(z_mu, z_sigma + self.eps).log_prob(z0).sum(-1) - log_j_q
        z_inv, log_j_p = self.prior_bijector.inverse(z)
        p_z = self.prior.log_prob(z_inv).sum(-1) + log_j_p
        return -(reconstruction_loss + p_z - q_z)