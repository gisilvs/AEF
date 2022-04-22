import torch
import torch.nn as nn
from torch.distributions import Normal


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
        else:
            deviations = None

        core, shell = self.forward(z, deviations)
        y = self.inverse_partition(core, shell)
        for i in range(len(self.preprocessing_layers) - 1, -1, -1):
            y, _ = self.preprocessing_layers[i](y)
        return y

    def forward(self, z, deviations=None):
        mu_d, sigma_d = self.decoder(z)
        if deviations is None:
            shell = mu_d
        else:
            shell = deviations * (sigma_d + self.eps) + mu_d
        shell = shell * (1 - self.mask)
        mu_z, sigma_z = self.encoder(shell)
        core = z * (sigma_z + self.eps) + mu_z
        core, _ = self.core_flow.forward(core)
        return core, shell
