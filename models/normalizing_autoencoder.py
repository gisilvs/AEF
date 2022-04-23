import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Normal
from torch.nn import ModuleList

from bijectors.realnvp import get_realnvp_bijector
from models.autoencoder import ConvolutionalEncoder, LatentDependentDecoder, GaussianEncoder, GaussianDecoder
from models.autoencoder_base import GaussianAutoEncoder


class NormalizingAutoEncoder(GaussianAutoEncoder):
    def __init__(self, encoder: GaussianEncoder, decoder: GaussianDecoder,
                 preprocessing_layers=[], hardcoded_mask=True):
        super(NormalizingAutoEncoder, self).__init__(encoder, decoder)


        self.core_size = self.encoder.latent_dim
        self.image_shape = self.encoder.input_shape
        self.core_flow_pre = get_realnvp_bijector(features=self.core_size,
                                                  hidden_features=256,
                                                  num_layers=4,
                                                  num_blocks_per_layer=2,
                                                  act_norm_between_layers=True)
        self.core_flow_post = get_realnvp_bijector(features=self.core_size,
                                                   hidden_features=256,
                                                   num_layers=4,
                                                   num_blocks_per_layer=2,
                                                   act_norm_between_layers=True)
        self.eps = 1e-5

        preprocessing_layers = nn.ModuleList(preprocessing_layers)
        self.preprocessing_layers = preprocessing_layers
        self.device = None

        if hardcoded_mask:
            mask = torch.zeros(self.image_shape)
            if self.core_size == 2:
                mask[0, 13:15, 13] = 1
            elif self.core_size == 4:
                mask[0, 13:15, 13:15] = 1
            elif self.core_size == 8:
                mask[0, 12:16, 13:15] = 1
            elif self.core_size == 16:
                mask[0, 12:16, 12:16] = 1
            else:
                print('NOT IMPLEMENTED YET')
                exit(1)
            self.register_buffer('mask', mask)
            #self.mask = mask
        else:
            mask = self._get_mask()
            self.register_buffer('mask', mask)
            #self.mask = self._get_mask()

    def _get_mask(self):
        '''
        simple procedure to fill up first each corner of each channel, and then proceeding along the sides.
        plot the mask for a visual understanding
        :return:
        '''
        mask = torch.zeros(self.image_shape)
        width = self.image_shape[1]
        height = self.image_shape[2]
        n_channels = self.image_shape[0]
        counter = 0
        row = 0
        column = 0
        channel = 0
        base_number_cols = 0
        base_number_rows = 0
        while (1):
            mask[channel, row, column] = 1
            counter += 1
            if counter == self.core_size:
                break
            mask[channel, row, height - column - 1] = 1
            counter += 1
            if counter == self.core_size:
                break
            mask[channel, width - row - 1, column] = 1
            counter += 1
            if counter == self.core_size:
                break
            mask[channel, width - row - 1, height - column - 1] = 1
            counter += 1
            if counter == self.core_size:
                break
            channel += 1
            if channel == n_channels:
                channel = 0
                if row == column:
                    row += 1
                elif row > column:
                    column = row
                    row = base_number_rows
                    if column > height // 2:
                        base_number_cols += 1
                        column = 0
                elif column > row:
                    row = column + 1
                    column = base_number_cols
                    if row > width // 2:
                        base_number_rows += 1
                        base_number_cols += 1
                        row = 0
        return mask

    def partition(self, x):
        core = x[:, self.mask == 1].view(x.shape[0], -1)
        shell = x * (1 - self.mask)
        return core, shell

    def inverse_partition(self, core, shell):
        shell[:, self.mask == 1] = core
        return shell

    def embedding(self, x):
        log_j_preprocessing = 0
        for layer in self.preprocessing_layers:
            x, log_j_transform = layer.inverse(x)
            log_j_preprocessing += log_j_transform

        core, shell = self.partition(x)
        core, log_j_core_pre = self.core_flow_pre.inverse(core)
        mu_z, sigma_z = self.encoder(shell)
        z = (core - mu_z) / (sigma_z + self.eps)
        log_j_z = torch.sum(-torch.log(sigma_z + self.eps), dim=[1])
        mu_d, sigma_d = self.decoder(z)
        deviations = (shell - mu_d) / (sigma_d + self.eps)
        log_j_d = torch.sum(-torch.log(sigma_d[:, self.mask == 0] + self.eps),
                            dim=[1])
        z, log_j_core_post = self.core_flow_post.inverse(z)
        return z, deviations, log_j_preprocessing + log_j_core_pre + log_j_z + log_j_d + log_j_core_post

    def neg_log_likelihood(self, x):
        z, deviations, log_j = self.embedding(x)
        loss_z = torch.sum(
            Normal(loc=torch.zeros_like(z), scale=torch.ones_like(z)).log_prob(
                z), dim=1)
        loss_d = torch.sum(Normal(loc=torch.zeros_like(deviations[:, self.mask == 0]),
                                  scale=torch.ones_like(deviations[:, self.mask == 0])).log_prob(
            deviations[:, self.mask == 0]), dim=[1])
        return -(loss_z + loss_d + log_j)

    def encode(self, x: Tensor):
        z, deviations, _ = self.embedding(x)
        return z, deviations

    def decode(self, z: Tensor, deviations: Tensor):
        core, shell = self.forward(z, deviations)
        x = self.inverse_partition(core, shell)
        return x

    def sample(self, num_samples=1, sample_deviations=False):
        device = self.get_device()
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
        z, _ = self.core_flow_post.forward(z)
        mu_d, sigma_d = self.decoder(z)
        if deviations is None:
            shell = mu_d
        else:
            shell = deviations * (sigma_d + self.eps) + mu_d
        shell = shell * (1 - self.mask)
        mu_z, sigma_z = self.encoder(shell)
        core = z * (sigma_z + self.eps) + mu_z
        core, _ = self.core_flow_pre.forward(core)
        return core, shell

    def loss_function(self, x: Tensor):
        return self.neg_log_likelihood(x)

    def get_device(self):
        if self.device is None:
            self.device = next(self.parameters()).device
            #self.mask = self.mask.to(self.device)
        return self.device
