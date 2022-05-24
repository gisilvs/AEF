import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Normal

from nflows.transforms import Transform

from models.autoencoder import GaussianEncoder, GaussianDecoder
from models.autoencoder_base import GaussianAutoEncoder


class InternalAEF(GaussianAutoEncoder):
    def __init__(self, encoder: GaussianEncoder, decoder: GaussianDecoder,
                 core_encoder: Transform, prior_flow: Transform,
                 mask: torch.Tensor, preprocessing_layers=[]):
        super(InternalAEF, self).__init__(encoder, decoder)

        self.core_size = self.encoder.latent_dim
        self.image_shape = self.encoder.input_shape
        self.core_encoder = core_encoder
        self.prior_flow = prior_flow
        self.eps = 1e-5

        #self.mask = mask
        self.preprocessing_layers = nn.ModuleList(preprocessing_layers)
        self.device = None

        self.register_buffer('mask', mask)  # Ensure mask gets sent to device

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
        core, log_j_core_pre = self.core_encoder.inverse(core)
        mu_z, sigma_z = self.encoder(shell)
        # Note: in the experiments the line below was implemented as 'z = (core - mu_z) / (sigma_z + self.eps)'
        # While different, we can convert between these two forms by taking mu_z_new = -mu_z_old/(sigma_z_old + eps)
        # and sigma_z_new = 1/(sigma_z_old + eps) - eps. In practice this makes no difference since the network will
        # just learn one or the other based on what is used for the implementation.
        z_1 = mu_z + (sigma_z + self.eps) * core
        log_j_z = torch.sum(torch.log(sigma_z + self.eps), dim=1)
        mu_d, sigma_d = self.decoder(z_1)
        deviations = (shell - mu_d) / (sigma_d + self.eps)
        log_j_d = torch.sum(-torch.log(sigma_d[:, self.mask == 0] + self.eps),
                            dim=1)
        z_0, log_j_core_post = self.prior_flow.inverse(z_1)
        return z_0, deviations, log_j_preprocessing + log_j_core_pre + log_j_z + log_j_d + log_j_core_post

    def neg_log_likelihood(self, x):
        z, deviations, log_j = self.embedding(x)
        loss_z = torch.sum(
            Normal(loc=torch.zeros_like(z), scale=torch.ones_like(z)).log_prob(
                z), dim=1)
        loss_d = torch.sum(Normal(loc=torch.zeros_like(deviations[:, self.mask == 0]),
                                  scale=torch.ones_like(deviations[:, self.mask == 0])).log_prob(
            deviations[:, self.mask == 0]), dim=1)
        return -(loss_z + loss_d + log_j)

    def encode(self, x: Tensor):
        z, deviations, _ = self.embedding(x)
        return z, deviations

    def decode(self, z_0: Tensor, deviations: Tensor = None):
        x = self.forward(z_0, deviations)
        return x

    def sample(self, num_samples: int = 1, sample_deviations: bool = False, temperature: float = 1, z_0: Tensor = None):
        device = self.get_device()

        deviations = None
        if z_0 is None:
            z_0 = torch.normal(torch.zeros(num_samples, self.core_size),
                               torch.ones(num_samples, self.core_size) * temperature).to(device)
            if sample_deviations:
                deviations = torch.normal(torch.zeros_like(self.mask),
                                          torch.ones_like(self.mask)*temperature).to(device)
        x = self.forward(z_0, deviations)
        return x

    def forward(self, z_0, deviations=None):
        z_1, _ = self.prior_flow.forward(z_0)
        mu_d, sigma_d = self.decoder(z_1)
        if deviations is None:
            shell = mu_d
        else:
            shell = deviations * (sigma_d + self.eps) + mu_d
        shell = shell * (1 - self.mask)
        mu_z, sigma_z = self.encoder(shell)
        core = (z_1 - mu_z) / (sigma_z + self.eps)
        core, _ = self.core_encoder.forward(core)
        x = self.inverse_partition(core, shell)
        for i in range(len(self.preprocessing_layers) - 1, -1, -1):
            x, _ = self.preprocessing_layers[i](x)

        return x

    def loss_function(self, x: Tensor):
        return self.neg_log_likelihood(x)

    def get_device(self):
        if self.device is None:
            self.device = next(self.parameters()).device
            # self.mask = self.mask.to(self.device)
        return self.device
