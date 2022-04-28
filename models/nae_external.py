import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Normal
from torch.nn import ModuleList
import numpy as np
from torch.nn.utils import weight_norm
from nflows.transforms.standard import IdentityTransform

from bijectors.realnvp import get_realnvp_bijector
from bijectors.masked_autoregressive_transform import get_masked_autoregressive_transform
from models.autoencoder import ConvolutionalEncoder, LatentDependentDecoder, GaussianEncoder, GaussianDecoder
from models.autoencoder_base import GaussianAutoEncoder

class MLP(nn.Module):
    '''
      Multilayer Perceptron.
    '''
    def __init__(self, input_size, latent_dim):
        super().__init__()
        self.layers = nn.Sequential(
          nn.Flatten(),
          weight_norm(nn.Linear(input_size, latent_dim)),)
    '''nn.ReLU(),
      weight_norm(nn.Linear(256, 256)),
      nn.ReLU(),
      weight_norm(nn.Linear(256, latent_dim))
    )'''
    '''def __init__(self, input_size, latent_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            weight_norm(nn.Linear(input_size, 256)),
            nn.Tanh(),
            weight_norm(nn.Linear(256, latent_dim)),
        )'''


    def forward(self, x):
      '''Forward pass'''
      return self.layers(x)

class NaeExternal(GaussianAutoEncoder):
    def __init__(self, encoder: GaussianEncoder, decoder: GaussianDecoder,
                 preprocessing_layers=[], core_flow_fn=get_masked_autoregressive_transform):
        super(NaeExternal, self).__init__(encoder, decoder)


        self.core_size = self.encoder.latent_dim
        self.image_shape = self.encoder.input_shape
        self.core_flow_pre = core_flow_fn(features=self.core_size,
                                                  hidden_features=256,
                                                  num_layers=4,
                                                  num_blocks_per_layer=2,
                                                  act_norm_between_layers=True)
        self.core_flow_pre = IdentityTransform()
        self.core_flow_post = core_flow_fn(features=self.core_size,
                                                   hidden_features=256,
                                                   num_layers=4,
                                                   num_blocks_per_layer=2,
                                                   act_norm_between_layers=True)
        self.eps = 1e-5
        preprocessing_layers = nn.ModuleList(preprocessing_layers)
        self.preprocessing_layers = preprocessing_layers
        self.device = None
        self.dense = MLP(np.prod(self.image_shape),self.core_size)

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
                            dim=[1,2,3])
        z, log_j_core_post = self.core_flow_post.inverse(z)
        return z, deviations, log_j_preprocessing + log_j_core_pre + log_j_z + log_j_d + log_j_core_post

    def neg_log_likelihood(self, x):
        z, deviations, log_j = self.embedding(x)
        loss_z = torch.sum(
            Normal(loc=torch.zeros_like(z), scale=torch.ones_like(z)).log_prob(
                z), dim=1)
        loss_d = torch.sum(Normal(loc=torch.zeros_like(deviations),
                                  scale=torch.ones_like(deviations)).log_prob(
            deviations), dim=[1,2,3])
        return -(loss_z + loss_d + log_j)

    def sample(self, num_samples=1, sample_deviations=False, temperature=1.):
        device = self.get_device()
        z = torch.normal(torch.zeros(num_samples, self.core_size),
                         torch.ones(num_samples, self.core_size)*temperature).to(device)
        if sample_deviations:
            deviations = torch.normal(torch.zeros_like(self.mask),
                                      torch.ones_like(self.mask)*temperature).to(device)
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
            #self.mask = self.mask.to(self.device)
        return self.device
