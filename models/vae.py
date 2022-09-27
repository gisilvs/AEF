from typing import List

import torch
from nflows.transforms import IdentityTransform, Transform
from torch import Tensor, distributions
from torch.distributions import Normal

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
                 posterior_bijector: Transform = IdentityTransform(), prior_bijector: Transform = IdentityTransform(),
                 preprocessing_layers=()):
        super(ExtendedVAE, self).__init__(encoder, decoder, posterior_bijector, prior_bijector, preprocessing_layers)

    def forward(self, x: Tensor):
        log_j_preprocessing = 0
        for layer in self.preprocessing_layers:
            x, log_j_transform = layer.inverse(x)
            log_j_preprocessing += log_j_transform

        z_mu, z_sigma = self.encoder(x)
        z0 = distributions.normal.Normal(z_mu, z_sigma + self.eps).rsample()
        z, log_j_q = self.posterior_bijector.forward(z0)
        x_mu, x_sigma = self.decoder(z)
        return x_mu, x_sigma, z_mu, z_sigma, z0, log_j_q, z, log_j_preprocessing, x

    def loss_function(self, x: Tensor):
        x_mu, x_sigma, z_mu, z_sigma, z0, log_j_q, z, log_j_preprocessing, x_preprocessed = self.forward(x)
        reconstruction_loss = torch.distributions.normal.Normal(x_mu, x_sigma + self.eps).log_prob(x_preprocessed).sum([1, 2, 3])
        q_z = distributions.normal.Normal(z_mu, z_sigma + self.eps).log_prob(z0).sum(-1) - log_j_q
        z_inv, log_j_p = self.prior_bijector.inverse(z)
        p_z = self.prior.log_prob(z_inv).sum(-1) + log_j_p
        return -(reconstruction_loss + p_z - q_z) - log_j_preprocessing

    def approximate_marginal(self, x: Tensor, n_samples: int = 128):
        # TODO: move towards implementations of base class
        batch_size = x.shape[0]

        log_j_preprocessing = 0
        for layer in self.preprocessing_layers:
            x, log_j_transform = layer.inverse(x)
            log_j_preprocessing += log_j_transform

        mu_z, sigma_z = self.encoder(x)
        z0_samples = Normal(mu_z, sigma_z).sample([n_samples]).transpose(1, 0)
        z_samples, log_j_posterior = self.posterior_bijector.forward(z0_samples.reshape(batch_size * n_samples, -1))
        mu_x, sigma_x = self.decoder(z_samples)

        mu_x, sigma_x = mu_x.view(batch_size, n_samples, -1), sigma_x.view(batch_size, n_samples, -1)
        p_x_z = Normal(mu_x, sigma_x).log_prob(x.view(batch_size, 1, -1)).sum([2]).view(batch_size, n_samples)
        z_prior, log_j_z_prior = self.prior_bijector.inverse(z_samples)
        #z_samples = z_samples.view(batch_size, n_samples, -1)
        log_j_posterior = log_j_posterior.view(batch_size, n_samples)
        p_latent = Normal(0, 1).log_prob(z_prior).sum([-1]) + log_j_z_prior
        p_latent = p_latent.view(batch_size, n_samples)
        q_latent = Normal(mu_z.unsqueeze(1), sigma_z.unsqueeze(1)).log_prob(z0_samples).sum([-1]) - log_j_posterior

        return -(torch.mean(torch.logsumexp(p_x_z + p_latent - q_latent, [1]) - torch.log(torch.tensor(n_samples)))) - log_j_preprocessing
