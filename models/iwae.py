import torch
from torch import Tensor, distributions, nn
from torch.distributions import Normal

from models.autoencoder import GaussianEncoder, GaussianDecoder
from models.autoencoder_base import GaussianAutoEncoder


class IWAE(GaussianAutoEncoder):
    def __init__(self, encoder: GaussianEncoder, decoder: GaussianDecoder, num_samples: int = 10):
        super().__init__(encoder, decoder)
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