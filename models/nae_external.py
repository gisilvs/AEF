import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Normal

import numpy as np

from nflows.transforms import Transform

from models.autoencoder import GaussianEncoder, GaussianDecoder
from models.autoencoder_base import GaussianAutoEncoder


class ExternalLatentAutoEncoder(GaussianAutoEncoder):
    def __init__(self, encoder: GaussianEncoder, decoder: GaussianDecoder,
                 core_flow_pre: Transform, core_flow_post: Transform,
                 external_net: nn.Module = None, preprocessing_layers=[]):
        super(ExternalLatentAutoEncoder, self).__init__(encoder, decoder)

        self.core_size = self.encoder.latent_dim
        self.image_shape = self.encoder.input_shape
        self.core_flow_pre = core_flow_pre
        self.core_flow_post = core_flow_post
        self.eps = 1e-5
        self.preprocessing_layers = nn.ModuleList(preprocessing_layers)
        self.device = None
        if external_net is None:
            self.dense = nn.Sequential(
          nn.Flatten(),
          nn.Linear(np.prod(self.image_shape), self.core_size))
        else:
            self.dense = external_net

    def embedding(self, x):
        log_j_preprocessing = 0
        for layer in self.preprocessing_layers:
            x, log_j_transform = layer.inverse(x)
            log_j_preprocessing += log_j_transform

        core = self.dense(x)
        core, log_j_core_pre = self.core_flow_pre.inverse(core)
        mu_z, sigma_z = self.encoder(x)
        z = (core - mu_z) / (sigma_z + self.eps)
        log_j_z = torch.sum(-torch.log(sigma_z + self.eps), dim=[1])
        mu_d, sigma_d = self.decoder(z)
        deviations = (x - mu_d) / (sigma_d + self.eps)
        log_j_d = torch.sum(-torch.log(sigma_d + self.eps),
                            dim=[1, 2, 3])
        z, log_j_core_post = self.core_flow_post.inverse(z)
        return z, deviations, log_j_preprocessing + log_j_core_pre + log_j_z + log_j_d + log_j_core_post

    def neg_log_likelihood(self, x):
        z, deviations, log_j = self.embedding(x)
        loss_z = torch.sum(
            Normal(loc=torch.zeros_like(z), scale=torch.ones_like(z)).log_prob(
                z), dim=1)
        loss_d = torch.sum(Normal(loc=torch.zeros_like(deviations),
                                  scale=torch.ones_like(deviations)).log_prob(
            deviations), dim=[1, 2, 3])
        return -(loss_z + loss_d + log_j)

    def encode(self, x: Tensor):
        z, deviations, _ = self.embedding(x)
        return z, deviations

    def decode(self, z: Tensor, deviations: Tensor = None):
        x = self.forward(z, deviations)
        return x

    def sample(self, num_samples=1, sample_deviations=False, temperature=1., z: Tensor = None):
        device = self.get_device()
        if z is None:
            z = torch.normal(torch.zeros(num_samples, self.core_size),
                             torch.ones(num_samples, self.core_size) * temperature).to(device)
        if sample_deviations:
            deviations = torch.normal(torch.zeros_like(self.mask),
                                      torch.ones_like(self.mask) * temperature).to(device)
        else:
            deviations = None

        y = self.forward(z, deviations)
        for i in range(len(self.preprocessing_layers) - 1, -1, -1):
            y, _ = self.preprocessing_layers[i](y)
        return y

    def forward(self, z, deviations=None):
        z, _ = self.core_flow_post.forward(z)
        mu_d, sigma_d = self.decoder(z)
        if deviations is None:
            x = mu_d
        else:
            x = deviations * (sigma_d + self.eps) + mu_d
        return x

    def loss_function(self, x: Tensor):
        return self.neg_log_likelihood(x)

    def get_device(self):
        if self.device is None:
            self.device = next(self.parameters()).device

        return self.device
