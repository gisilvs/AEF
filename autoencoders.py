import torch
import torch.nn as nn
from torch.distributions import Normal
import torch.nn.functional as F


class NormalizingAutoEncoder(nn.Module):
    def __init__(self, core_flow, encoder, decoder, mask, preprocessing_layers=[]):
        super().__init__()

        self.core_flow = core_flow
        self.encoder = encoder
        self.decoder = decoder
        self.eps = 1e-5
        self.core_size = int(torch.sum(mask))
        self.mask = mask
        self.preprocessing_layers = preprocessing_layers

    def partition(self, x):
        core = x[:, :, self.mask == 1].view(x.shape[0], -1)
        shell = x * (1 - self.mask)
        return core, shell

    def inverse_partition(self, core, shell):
        shell[:, :, self.mask == 1] = core.reshape(shell.shape[0], -1,
                                              self.core_size)
        return shell

    def embedding(self, x):
        log_j_preprocessing = 0
        for layer in self.preprocessing_layers:
            x, log_j_transform = layer.inverse(x)
            log_j_preprocessing += log_j_transform

        core, shell = self.partition(x)
        core, log_j_core = self.core_flow.inverse(core)
        mu_z, sigma_z = self.encoder(shell)
        z = (core - mu_z) / (sigma_z + self.eps)
        log_j_z = torch.sum(-torch.log(sigma_z + self.eps), dim=[1])
        mu_d, sigma_d = self.decoder(z)
        deviations = (shell - mu_d) / (sigma_d + self.eps)
        log_j_d = torch.sum(-torch.log(sigma_d[:, :, self.mask == 0] + self.eps),
                            dim=[1, 2])
        return z, deviations, log_j_preprocessing + log_j_core + log_j_z + log_j_d

    def neg_log_likelihood(self, x):
        z, deviations, log_j = self.embedding(x)
        loss_z = torch.sum(
            Normal(loc=torch.zeros_like(z), scale=torch.ones_like(z)).log_prob(
                z), dim=1)
        loss_d = torch.sum(Normal(loc=torch.zeros_like(deviations[:, :, self.mask == 0]),
                                  scale=torch.ones_like(deviations[:, :, self.mask == 0])).log_prob(
            deviations[:, :, self.mask == 0]), dim=[1, 2])
        return -(loss_z + loss_d + log_j)

    def sample(self, num_samples=1, sample_deviations=False):
        device = next(self.parameters()).device
        z = torch.normal(torch.zeros(num_samples, self.core_size),
                         torch.ones(num_samples, self.core_size)).to(device)

        if sample_deviations:
            deviations = torch.normal(torch.zeros_like(self.mask),
                                      torch.ones_like(self.mask)).to(device)
            core, shell = self.forward(z, deviations)
        else:
            shell, _ = self.decoder(z)
            shell = shell * (1 - self.mask)
            mu_z, sigma_z = self.encoder(shell)
            core = z * (sigma_z + self.eps) + mu_z
            core = self.core_flow.forward(core)
        y = self.inverse_partition(core, shell)
        for i in range(len(self.preprocessing_layers) - 1, -1, -1):
            y, _ = self.preprocessing_layers[i](y)
        return y

    def forward(self, z, deviations):
        mu_d, sigma_d = self.decoder(z)
        shell = deviations * (sigma_d + self.eps) + mu_d
        shell = shell * (1 - self.mask)
        mu_z, sigma_z = self.encoder(shell)
        core = z * (sigma_z + self.eps) + mu_z
        core = self.core_flow.forward(core)
        return core, shell



class Encoder(nn.Module):
    def __init__(self, hidden_channels: int, latent_dim: int, input_channels: int = 1):
        """
        Simple encoder module

        It predicts the `mean` and `log(variance)` parameters.

        The choice to use the `log(variance)` is for stability reasons:
        https://stats.stackexchange.com/a/353222/284141
        """
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=input_channels,
                               out_channels=hidden_channels,
                               kernel_size=3,
                               stride=2,
                               padding=1)  # out: hidden_channels x 14 x 14

        self.conv2 = nn.Conv2d(in_channels=hidden_channels,
                               out_channels=hidden_channels * 2,
                               kernel_size=3,
                               stride=2,
                               padding=1)  # out: (hidden_channels x 2) x 7 x 7

        self.fc_mu = nn.Linear(in_features=hidden_channels * 2 * 7 * 7,
                               out_features=latent_dim)
        self.fc_logvar = nn.Linear(in_features=hidden_channels * 2 * 7 * 7,
                                   out_features=latent_dim)

        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor):
        """
        :param x: batch of images with shape [batch, channels, w, h]
        :returns: the predicted mean and log(variance)
        """
        batch_size = x.shape[0]
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))

        x = x.view(x.shape[0], -1)

        x_mu = self.fc_mu(x)
        x_mu = x_mu
        x_logvar = self.fc_logvar(x)
        x_logvar = x_logvar

        return x_mu.view(batch_size, -1), F.softplus(x_logvar).view(batch_size,
                                                                    -1)


class Decoder(nn.Module):
    def __init__(self, hidden_channels, latent_dim, output_shape):
        """
        Simple decoder module
        """
        super().__init__()

        self.output_shape = output_shape
        self.hidden_channels = hidden_channels

        self.fc = nn.Linear(in_features=latent_dim,
                            out_features=hidden_channels * 2 * 7 * 7)

        self.conv2 = nn.ConvTranspose2d(in_channels=hidden_channels * 2,
                                        out_channels=hidden_channels,
                                        kernel_size=4,
                                        stride=2,
                                        padding=1)
        self.conv1 = nn.ConvTranspose2d(in_channels=hidden_channels,
                                        out_channels=output_shape[0],
                                        kernel_size=4,
                                        stride=2,
                                        padding=1)

        self.activation = nn.ReLU()
        self.pre_sigma = nn.Parameter(torch.zeros(output_shape))

    def forward(self, x: torch.Tensor):
        """
        :param x: a sample from the distribution governed by the mean and log(var)
        :returns: a reconstructed image with size [batch, 1, w, h]
        """
        x = self.fc(x)
        x = x.view(x.size(0), self.hidden_channels * 2, 7, 7)
        x = self.activation(self.conv2(x))
        x = self.conv1(x)
        return x, F.softplus(self.pre_sigma).unsqueeze(0).repeat(x.shape[0],1,1,1)

