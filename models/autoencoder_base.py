import torch
from torch import nn, Tensor, distributions
from nflows.transforms import Transform, IdentityTransform

from models.autoencoder import GaussianEncoder, GaussianDecoder

from torch.distributions.normal import Normal


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

    def encode(self, x: Tensor):
        raise NotImplementedError

    def decode(self, z: Tensor):
        raise NotImplementedError

    def sample(self, n_samples: int):
        raise NotImplementedError

    def forward(self, x: Tensor):
        raise NotImplementedError

    def loss_function(self, x: Tensor):
        raise NotImplementedError


class GaussianAutoEncoder(AutoEncoder):
    def __init__(self, encoder: GaussianEncoder, decoder: GaussianDecoder):
        super(GaussianAutoEncoder, self).__init__()

        assert encoder.latent_dim == decoder.latent_dim

        self.encoder = encoder
        self.decoder = decoder
        self.latent_dims = encoder.latent_dim
        self.prior = Normal(torch.zeros(self.latent_dims),
                                        torch.ones(self.latent_dims))
        self.eps = 1e-6
        self.device = None

    def encode(self, x: Tensor):
        return self.encoder(x)

    def decode(self, z: Tensor):
        return self.decoder(z)

    def sample(self, num_samples: int = 1, temperature: int = 1):
        self.set_device()

        z = self.prior.sample((num_samples, )) * temperature
        return self.decode(z)[0]

    def approximate_marginal(self, images: Tensor, n_samples: int = 128):
        batch_size = images.shape[0]
        mu_z, sigma_z = self.encoder(images)
        samples = Normal(mu_z, sigma_z).sample([n_samples]).transpose(1, 0)
        mu_x, sigma_x = self.decoder(samples.reshape(batch_size * n_samples, -1))
        mu_x, sigma_x = mu_x.view(batch_size, n_samples, -1), sigma_x.view(batch_size, n_samples, -1)
        p_x_z = Normal(mu_x, sigma_x).log_prob(images.view(batch_size, 1, -1)).sum([2]).view(batch_size, n_samples)
        p_latent = Normal(0, 1).log_prob(samples).sum([-1])
        q_latent = Normal(mu_z.unsqueeze(1), sigma_z.unsqueeze(1)).log_prob(samples).sum([-1])

        return torch.mean(torch.logsumexp(p_x_z + p_latent - q_latent, [1]) - torch.log(torch.tensor(n_samples)))

    def set_device(self):
        # TODO: override to? see https://stackoverflow.com/questions/59179609/how-to-make-a-pytorch-distribution-on-gpu
        if self.device is None:
            self.device = next(self.encoder.parameters()).device
            # Putting loc and scale to device gives nans for some reason
            self.prior = distributions.normal.Normal(torch.zeros(self.latent_dims).to(self.device),
                                                     torch.ones(self.latent_dims).to(self.device))


class ExtendedGaussianAutoEncoder(GaussianAutoEncoder):
    def __init__(self, encoder: GaussianEncoder, decoder: GaussianDecoder, 
                 posterior_bijector: Transform = IdentityTransform(), prior_bijector: Transform = IdentityTransform()):
        super(ExtendedGaussianAutoEncoder, self).__init__(encoder, decoder)
        self.posterior_bijector = posterior_bijector
        self.prior_bijector = prior_bijector

    def sample(self, num_samples: int = 1, temperature: float = 1):
        self.set_device()
        z0 = self.prior.sample((num_samples,)) * temperature
        z, _ = self.prior_bijector.forward(z0)
        return self.decoder(z)[0]

    def encode(self, x: Tensor):
        z0 = self.encoder(x)[0]
        z = self.posterior_bijector.forward(z0)[0]
        return z

    def approximate_marginal(self, images: Tensor, n_samples: int = 128):
        batch_size = images.shape[0]
        mu_z, sigma_z = self.encoder(images)
        z0_samples = Normal(mu_z, sigma_z).sample([n_samples]).transpose(1, 0)
        z_samples, log_j_posterior = self.posterior_bijector.forward(z0_samples.reshape(batch_size * n_samples, -1))
        mu_x, sigma_x = self.decoder(z_samples)
        z_samples = z_samples.view(batch_size, n_samples, -1)
        mu_x, sigma_x = mu_x.view(batch_size, n_samples, -1), sigma_x.view(batch_size, n_samples, -1)
        p_x_z = Normal(mu_x, sigma_x).log_prob(images.view(batch_size, 1, -1)).sum([2]).view(batch_size, n_samples)
        z_prior, log_j_z_prior = self.prior_bijector.inverse(z_samples)
        p_latent = Normal(0, 1).log_prob(z_prior).sum([-1]) + log_j_z_prior
        q_latent = Normal(mu_z.unsqueeze(1), sigma_z.unsqueeze(1)).log_prob(z0_samples).sum([-1]) - log_j_posterior

        return torch.mean(torch.logsumexp(p_x_z + p_latent - q_latent, [1]) - torch.log(torch.tensor(n_samples)))