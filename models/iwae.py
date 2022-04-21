import torch
from torch import Tensor, distributions, nn
from torch.distributions import Normal

from models.autoencoder_base import AutoEncoder
from models.autoencoder import Encoder, IndependentVarianceDecoder
import torch.nn.functional as F
from models.vae import VAE

class IWAE(AutoEncoder):
    def __init__(self, hidden_channels: int, latent_dim: int, input_dim: torch.Size, num_samples: int = 10):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.eps = 1e-5
        self.num_samples = num_samples

        self.encoder = Encoder(hidden_channels, latent_dim, input_dim)
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
        batch_size = x.shape[0]
        mu_z, sigma_z = self.encode(x)
        samples = Normal(mu_z, sigma_z).rsample([self.num_samples]).transpose(1, 0)
        mu_x, sigma_x = self.decode(samples.reshape(batch_size * self.num_samples, -1))
        mu_x, sigma_x = mu_x.view(batch_size, self.num_samples, -1), sigma_x.view(batch_size, self.num_samples, -1)
        p_x_z = Normal(mu_x, sigma_x).log_prob(x.view(batch_size, 1, -1)).sum([2]).view(batch_size, self.num_samples)
        p_latent = Normal(0, 1).log_prob(samples).sum([-1])
        q_latent = Normal(mu_z.unsqueeze(1), sigma_z.unsqueeze(1)).log_prob(samples).sum([-1])

        return -torch.mean(torch.logsumexp(p_x_z + p_latent - q_latent, [1]) - torch.log(torch.tensor(self.num_samples)))

    def get_device(self):
        if self.device is None:
            self.device = next(self.encoder.parameters()).device
            # Putting loc and scale to device gives nans for some reason
            self.prior = distributions.normal.Normal(torch.zeros(self.latent_dim).to(self.device),
                                                     torch.ones(self.latent_dim).to(self.device))
        return self.device
