import torch
from torch import Tensor, distributions, nn
from autoencoder_base import AutoEncoder
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
        z = distributions.normal.Normal(z_mu, z_sigma).rsample((self.num_samples,)).to(self.get_device())
        z = z.permute(1, 0, 2)  # [batch_samples, h_samples, latent_space]
        z = z.reshape(-1, z.shape[2])
        x_mu, x_sigma = self.decode(z)
        x_mu = x_mu.view([x.shape[0], self.num_samples, x.shape[1], x.shape[2], x.shape[3]])  # [B,H,C,H,W]
        x_sigma = x_sigma.view([x.shape[0], self.num_samples, x.shape[1], x.shape[2], x.shape[3]])
        return x_mu, x_sigma, z_mu, z_sigma, z

    def loss_function(self, x: Tensor):
        x_mu, x_sigma, z_mu, z_sigma, z = self.forward(x)
        # reconstruction_loss = torch.distributions.normal.Normal(x_mu,x_sigma+self.eps).log_prob(x).sum([1,2,3])

        # Manual
        # Log p(z)
        log_p_z = self.prior.log_prob(z).sum(-1)
        log_p_z = log_p_z.reshape(x.shape[0], self.num_samples)
        # Log p(x|z)
        x_ = x.unsqueeze(1).repeat(1, self.num_samples, 1, 1, 1)
        log_p_x_z = distributions.normal.Normal(x_mu, x_sigma+self.eps).log_prob(x_).sum([2, 3, 4])

        # Log P(z|x)
        z_mu_ = z_mu.repeat(self.num_samples, 1)
        z_sigma_ = z_sigma.repeat(self.num_samples, 1)
        log_q_z_x = distributions.normal.Normal(z_mu_, z_sigma_+self.eps).log_prob(z).sum(-1)
        log_q_z_x = log_q_z_x.reshape(x.shape[0], self.num_samples)
        q_z_x = distributions.normal.Normal(z_mu, z_sigma)
        kl_div = distributions.kl.kl_divergence(q_z_x, self.prior).sum(-1)
        old = log_q_z_x - log_p_z


        log_w = log_p_x_z - kl_div.unsqueeze(1) #+ log_p_z - log_q_z_x
        log_w_ = log_w - log_w.max(dim=1)[0].unsqueeze(1)
        w_normalized_ = F.softmax(log_w_, dim=1)

        loss = -torch.sum(log_w * w_normalized_, dim=1)
        return loss

        # q_z = distributions.normal.Normal(z_mu, z_sigma)
        # p_z = distributions.normal.Normal(torch.zeros(self.latent_dim).to(self.get_device()),
        #                                   torch.ones(self.latent_dim).to(self.get_device()))
        #
        # kl_div = distributions.kl.kl_divergence(q_z,p_z).sum(1)
        # return reconstruction_loss + kl_div

    def get_device(self):
        if self.device is None:
            self.device = next(self.encoder.parameters()).device
            # Putting loc and scale to device gives nans for some reason
            self.prior = distributions.normal.Normal(torch.zeros(self.latent_dim).to(self.device),
                                                     torch.ones(self.latent_dim).to(self.device))
        return self.device
