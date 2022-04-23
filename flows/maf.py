from bijectors.masked_autoregressive_transform import get_masked_autoregressive_transform
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
class MaskedAutoregressiveFlow(nn.Module):
    def __init__(self,
                 features,
                 hidden_features,
                 num_layers,
                 num_blocks_per_layer,
                 preprocessing_layers=[],
                 use_residual_blocks=True,
                 use_random_masks=False,
                 use_random_permutations=False,
                 activation=F.relu,
                 dropout_probability=0.0,
                 batch_norm_within_layers=False,
                 act_norm_between_layers=False):
        super(MaskedAutoregressiveFlow, self).__init__()

        self.features = features
        self.base_dist = Normal(torch.zeros(self.features),torch.ones(self.features))
        self.maf = get_masked_autoregressive_transform(features,
                                                       hidden_features,
                                                       num_layers,
                                                       num_blocks_per_layer,
                                                       use_residual_blocks,
                                                       use_random_masks,
                                                       use_random_permutations,
                                                       activation,
                                                       dropout_probability,
                                                       batch_norm_within_layers,
                                                       act_norm_between_layers)

        self.preprocessing_layers = nn.ModuleList(preprocessing_layers)

    def forward(self, x):
        y, log_j_maf = self.maf.forward(x)
        log_j_preprocessing = 0
        for i in range(len(self.preprocessing_layers) - 1, -1, -1):
            y, log_j_transform = self.preprocessing_layers[i](y)
            log_j_preprocessing += log_j_transform
        return y, log_j_maf + log_j_preprocessing

    def inverse(self, y):
        log_j_preprocessing = 0
        for layer in self.preprocessing_layers:
            y, log_j_transform = layer.inverse(y)
            log_j_preprocessing += log_j_transform
        x, log_j_maf = self.maf.inverse(y)
        return x, log_j_maf + log_j_preprocessing

    def sample(self, n_samples):
        samples = self.base_dist.sample([n_samples])
        return self.forward(samples)[0]

    def loss_function(self, x):
        z, log_j = self.inverse(x)
        return -(self.base_dist.log_prob(z).sum(-1) + log_j)
