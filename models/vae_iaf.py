import torch
from torch import Tensor, distributions

from bijectors.masked_autoregressive_transform import get_masked_autoregressive_transform
from models.autoencoder import GaussianEncoder, GaussianDecoder
from models.autoencoder_base import GaussianAutoEncoder
from models.vae import ExtendedVAE


class VAEIAF(GaussianAutoEncoder):
    def __init__(self, encoder: GaussianEncoder, decoder: GaussianDecoder):
        super(VAEIAF, self).__init__(encoder, decoder)
        self.iaf = get_masked_autoregressive_transform(self.latent_dims, 256, 8, 2, act_norm_between_layers=True, is_inverse=True)

    def forward(self, x: Tensor):
        self.set_device()
        z_mu, z_sigma = self.encoder(x)
        z0 = distributions.normal.Normal(z_mu, z_sigma + self.eps).rsample()
        z, log_j_z = self.iaf.forward(z0)
        x_mu, x_sigma = self.decoder(z)
        return x_mu, x_sigma, z_mu, z_sigma, z0, log_j_z, z

    def encode(self, x: Tensor):
        z0 = self.encoder(x)[0]
        z = self.iaf.forward(z0)
        return z

    def loss_function(self, x: Tensor):
        self.set_device()
        x_mu, x_sigma, z_mu, z_sigma, z0, log_j_z, z = self.forward(x)
        reconstruction_loss = torch.distributions.normal.Normal(x_mu, x_sigma + self.eps).log_prob(x).sum([1, 2, 3])
        q_z = distributions.normal.Normal(z_mu, z_sigma + self.eps).log_prob(z0).sum(-1) - log_j_z
        p_z = distributions.normal.Normal(0, 1).log_prob(z).sum(-1)
        return -(reconstruction_loss + p_z - q_z)

    def approximate_marginal(self, images: Tensor, n_samples: int = 128):
    #     batch_size = x.shape[0]
    #     z_mu, z_sigma = self.encoder(x)
    #     z = Normal(z_mu, z_sigma + self.eps).rsample([self.num_samples]).transpose(1, 0)
    #     x_mu, x_sigma = self.decoder(z.reshape(batch_size * self.num_samples, -1))
    #     x_mu, x_sigma = x_mu.view(batch_size, self.num_samples, -1), x_sigma.view(batch_size, self.num_samples, -1)
    #     return x_mu, x_sigma, z_mu, z_sigma, z
    #
    # def loss_function(self, x: Tensor):
    #     self.set_device()
    #     batch_size = x.shape[0]
    #     x_mu, x_sigma, z_mu, z_sigma, z = self.forward(x)
    #     p_x_z = Normal(x_mu, x_sigma + self.eps).log_prob(x.view(batch_size, 1, -1)).sum([2]).view(batch_size,
    #                                                                                                self.num_samples)
    #     p_latent = self.prior.log_prob(z).sum([-1])
    #     q_latent = Normal(z_mu.unsqueeze(1), z_sigma.unsqueeze(1) + self.eps).log_prob(z).sum([-1])
    #
    #     return -torch.mean(
    #         torch.logsumexp(p_x_z + p_latent - q_latent, [1]) - torch.log(torch.tensor(self.num_samples)))

        batch_size = images.shape[0]
        mu_z, sigma_z = self.encoder(images)
        z0_samples = distributions.normal.Normal(mu_z, sigma_z).sample([n_samples]).transpose(1, 0)
        z_samples, _ = self.iaf.forward(z0_samples.reshape(batch_size * n_samples, -1))
        mu_x, sigma_x = self.decoder(z_samples)
        z_samples = z_samples.view(batch_size, n_samples, -1)
        mu_x, sigma_x = mu_x.view(batch_size, n_samples, -1), sigma_x.view(batch_size, n_samples, -1)
        p_x_z = distributions.normal.Normal(mu_x, sigma_x).log_prob(images.view(batch_size, 1, -1)).sum([2]).view(batch_size, n_samples)
        p_latent = distributions.normal.Normal(0, 1).log_prob(z_samples).sum([-1])
        q_latent = distributions.normal.Normal(mu_z.unsqueeze(1), sigma_z.unsqueeze(1)).log_prob(z0_samples).sum([-1])

        return torch.mean(torch.logsumexp(p_x_z + p_latent - q_latent, [1]) - torch.log(torch.tensor(n_samples)))


class ExtendedVAEIAF(ExtendedVAE):
    def __init__(self, encoder: GaussianEncoder, decoder: GaussianDecoder, hidden_features: int = 256,
                 num_layers: int = 8, num_blocks_per_layer: int = 2):
        iaf = get_masked_autoregressive_transform(features=encoder.latent_dims,
                                                  hidden_features=hidden_features, num_layers=num_layers,
                                                  num_blocks_per_layer=num_blocks_per_layer,
                                                  act_norm_between_layers=True,
                                                  is_inverse=True)
        super(VAEIAF, self).__init__(encoder, decoder, posterior_bijector=iaf)


class DenoisingVAEIAF(VAEIAF):
    def __init__(self, encoder: GaussianEncoder, decoder: GaussianDecoder):
        super(DenoisingVAEIAF, self).__init__(encoder, decoder)

    def loss_function(self, x_noisy: Tensor, x_original: Tensor):
        x_mu, x_sigma, z_mu, z_sigma, z0, log_j_z, z = self.forward(x_noisy)
        reconstruction_loss = torch.distributions.normal.Normal(x_mu, x_sigma + self.eps).log_prob(x_original).sum([1, 2, 3])
        q_z = distributions.normal.Normal(z_mu, z_sigma + self.eps).log_prob(z0).sum(-1) - log_j_z
        p_z = distributions.normal.Normal(0, 1).log_prob(z).sum(-1)
        return -(reconstruction_loss + p_z - q_z)
