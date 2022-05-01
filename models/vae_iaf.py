import torch
from torch import Tensor, distributions

from bijectors.masked_autoregressive_transform import get_masked_autoregressive_transform
from models.autoencoder import GaussianEncoder, GaussianDecoder
from models.autoencoder_base import GaussianAutoEncoder
from models.vae import ExtendedVAE


class VAEIAF(ExtendedVAE):
    def __init__(self, encoder: GaussianEncoder, decoder: GaussianDecoder, hidden_features: int = 256,
                 num_layers: int = 8, num_blocks_per_layer: int = 2):
        iaf = get_masked_autoregressive_transform(features=encoder.latent_ndims,
                                                  hidden_features=hidden_features, num_layers=num_layers,
                                                  num_blocks_per_layer=num_blocks_per_layer,
                                                  act_norm_between_layers=True,
                                                  is_inverse=True)
        super(VAEIAF, self).__init__(encoder, decoder, posterior_bijector=iaf)
