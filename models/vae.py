import torch
from torch import Tensor, distributions, nn
from autoencoder_base import AutoEncoder
from models.autoencoder import Encoder, Decoder
import torch.nn.functional as F


class VAE(AutoEncoder):
    def __init__(self, hidden_channels: int, latent_dim: int, input_dim: torch.Size):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.latent_dim = latent_dim
        self.input_dim = input_dim

        self.encoder = Encoder(hidden_channels, latent_dim, input_dim)
        self.decoder = Decoder(hidden_channels, latent_dim, input_dim)
        self.decoder2 = Decoder(hidden_channels, latent_dim, input_dim)

        self.prior = distributions.normal.Normal(torch.zeros(latent_dim), torch.ones(latent_dim))
        self.device = None

    def encode(self, x: Tensor):
        return self.encoder(x) # Encoder returns mu and log(sigma)

    def decode(self, z: Tensor, sample_from_likelihood=False):
        decoded_mu, decoded_sigma = self.decoder(z)
        if sample_from_likelihood:
            pass # TODO: implement if needed
        else:
            return decoded_mu

    def sample(self, num_samples: int):
        z = self.prior.sample((num_samples, )).to(self.get_device())
        return self.decode(z)

    def forward(self, x: Tensor):
        z_mu, z_sigma = self.encode(x)
        z = distributions.normal.Normal(z_mu, z_sigma).rsample().to(self.get_device())
        x_mu, _ = self.decoder(z)
        _ , x_sigma = self.decoder2(z)
        return x_mu, x_sigma, z_mu, z_sigma

    def loss_function(self, x: Tensor):
        x_mu, x_sigma, z_mu, z_sigma = self.forward(x)

        reconstruction_loss = -torch.distributions.normal.Normal(x_mu,x_sigma+1e-5).log_prob(x).sum([1,2,3])

        #reconstruction_loss = nn.GaussianNLLLoss(full=True, reduction='none', eps=1e-4)(x, x_mu, x_sigma**2).sum([1,2,3])
        q_z = distributions.normal.Normal(z_mu, z_sigma)
        p_z = distributions.normal.Normal(torch.zeros(self.latent_dim).to(self.get_device()),
                                          torch.ones(self.latent_dim).to(self.get_device()))

        kl_div = distributions.kl.kl_divergence(q_z,p_z).sum(1)
        return reconstruction_loss + kl_div

    def get_device(self):
        if self.device is None:
            self.device = next(self.encoder.parameters()).device
        return self.device
