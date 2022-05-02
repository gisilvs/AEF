import torch
from nflows.transforms import Transform, IdentityTransform
from torch import Tensor, distributions, nn
from torch.distributions import Normal

from models.autoencoder import GaussianEncoder, GaussianDecoder
from models.autoencoder_base import GaussianAutoEncoder, ExtendedGaussianAutoEncoder


class IWAE(GaussianAutoEncoder):
    def __init__(self, encoder: GaussianEncoder, decoder: GaussianDecoder, num_samples: int = 10):
        super(IWAE, self).__init__(encoder, decoder)
        self.num_samples = num_samples

    def forward(self, x: Tensor):
        batch_size = x.shape[0]
        z_mu, z_sigma = self.encode(x)
        z = Normal(z_mu, z_sigma + self.eps).rsample([self.num_samples]).transpose(1, 0)
        x_mu, x_sigma = self.decode(z.reshape(batch_size * self.num_samples, -1))
        x_mu, x_sigma = x_mu.view(batch_size, self.num_samples, -1), x_sigma.view(batch_size, self.num_samples, -1)
        return x_mu, x_sigma, z_mu, z_sigma, z

    def loss_function(self, x: Tensor):
        self.set_device()
        batch_size = x.shape[0]
        x_mu, x_sigma, z_mu, z_sigma, z = self.forward(x)
        p_x_z = Normal(x_mu, x_sigma+self.eps).log_prob(x.view(batch_size, 1, -1)).sum([2]).view(batch_size, self.num_samples)
        p_latent = self.prior.log_prob(z).sum([-1])
        q_latent = Normal(z_mu.unsqueeze(1), z_sigma.unsqueeze(1)+self.eps).log_prob(z).sum([-1])

        return -torch.mean(torch.logsumexp(p_x_z + p_latent - q_latent, [1]) - torch.log(torch.tensor(self.num_samples)))


class ExtendedIWAE(ExtendedGaussianAutoEncoder):
    def __init__(self, encoder: GaussianEncoder, decoder: GaussianDecoder,
                 posterior_bijector: Transform = IdentityTransform(), prior_bijector: Transform = IdentityTransform(),
                 num_samples: int = 10):
        super(ExtendedIWAE, self).__init__(encoder, decoder, posterior_bijector, prior_bijector)
        self.num_samples = num_samples

    def forward(self, x: Tensor):
        self.set_device()
        batch_size = x.shape[0]
        z_mu, z_sigma = self.encode(x)
        z0 = distributions.normal.Normal(z_mu, z_sigma + self.eps).rsample([self.num_samples]).transpose(1, 0)
        z, log_j_q = self.posterior_bijector.forward(z0.reshape(batch_size * self.num_samples, -1))
        log_j_q = log_j_q.view(batch_size, self.num_samples)
        x_mu, x_sigma = self.decode(z)
        x_mu, x_sigma = x_mu.view(batch_size, self.num_samples, -1), x_sigma.view(batch_size, self.num_samples, -1)
        return x_mu, x_sigma, z_mu, z_sigma, z0, log_j_q, z

    def loss_function(self, x: Tensor):
        self.set_device()
        batch_size = x.shape[0]
        x_mu, x_sigma, z_mu, z_sigma, z0, log_j_q, z = self.forward(x)
        p_x_z = Normal(x_mu, x_sigma+self.eps).log_prob(x.view(batch_size, 1, -1)).sum([2]).view(batch_size, self.num_samples)
        q_z = distributions.normal.Normal(z_mu.unsqueeze(1), z_sigma.unsqueeze(1) + self.eps).log_prob(z0).sum(-1) - log_j_q
        z_inv, log_j_p = self.prior_bijector.inverse(z)
        p_z = self.prior.log_prob(z_inv).sum(-1) + log_j_p
        p_z = p_z.view(batch_size, self.num_samples)

        return -torch.mean(torch.logsumexp(p_x_z + p_z - q_z, [1]) - torch.log(torch.tensor(self.num_samples)))

class DenoisingIWAE(IWAE):
    def __init__(self, encoder: GaussianEncoder, decoder: GaussianDecoder, num_samples: int = 10):
        super(DenoisingIWAE, self).__init__(encoder, decoder)
        self.num_samples = num_samples

    def loss_function(self, x_noisy: Tensor, x_original: Tensor):
        self.set_device()
        batch_size = x_noisy.shape[0]
        x_mu, x_sigma, z_mu, z_sigma, z = self.forward(x_noisy)
        p_x_z = Normal(x_mu, x_sigma+self.eps).log_prob(x_original.view(batch_size, 1, -1)).sum([2]).view(batch_size, self.num_samples)
        p_latent = self.prior.log_prob(z).sum([-1])
        q_latent = Normal(z_mu.unsqueeze(1), z_sigma.unsqueeze(1)+self.eps).log_prob(z).sum([-1])

        return -torch.mean(torch.logsumexp(p_x_z + p_latent - q_latent, [1]) - torch.log(torch.tensor(self.num_samples)))