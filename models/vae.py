import torch
from torch import Tensor, distributions, nn
from autoencoder_base import AutoEncoder
from models.autoencoder import Encoder, Decoder


class VAE(AutoEncoder):
    def __init__(self, hidden_channels: int, latent_dim: int, input_dim: torch.Size):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.latent_dim = latent_dim
        self.input_dim = input_dim

        self.encoder = Encoder(hidden_channels, latent_dim, input_dim)
        self.decoder = Decoder(hidden_channels, latent_dim, input_dim)

        self.prior = distributions.normal.Normal(torch.zeros(latent_dim), torch.ones(latent_dim))
        self.device = None

    def encode(self, x: Tensor):
        return self.encoder(x) # Encoder returns mu and log(sigma)

    def decode(self, z: Tensor, sample_from_likelihood=False):
        decoded_mu, decoded_log_sigma = self.decoder(z)
        if sample_from_likelihood:
            pass # TODO: implement if needed
        else:
            return decoded_mu

    def sample(self, num_samples: int):
        z = self.prior.sample((num_samples, )).to(self.get_device())
        return self.decode(z)

    def forward(self, x: Tensor):
        z_mu, z_log_sigma = self.encode(x)
        z = distributions.normal.Normal(z_mu, torch.exp(z_log_sigma)).rsample().to(self.get_device())
        x_reconstructed = self.decode(z)
        return x_reconstructed, z_mu, z_log_sigma

    def loss_function(self, x: Tensor):
        x_reconstructed, z_mu, z_log_sigma = self.forward(x)
        reconstruction_loss_func = nn.MSELoss()
        reconstruction_loss = reconstruction_loss_func(x, x_reconstructed)


        q_z = distributions.normal.Normal(z_mu, torch.exp(z_log_sigma))
        p_z = distributions.normal.Normal(torch.zeros(self.latent_dim).to(self.get_device()),
                                          torch.ones(self.latent_dim).to(self.get_device()))
        kl_div = distributions.kl.kl_divergence(p_z, q_z)
        return reconstruction_loss + kl_div

    def get_device(self):
        if self.device is None:
            self.device = next(self.encoder.parameters()).device
        return self.device
