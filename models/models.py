from typing import List

import torch
from nflows.transforms import InverseTransform, AffineTransform

from bijectors.actnorm import ActNorm
from bijectors.sigmoid import Sigmoid
from flows.maf import MaskedAutoregressiveFlow
from .autoencoder import IndependentVarianceDecoder, LatentDependentDecoder, ConvolutionalEncoder, \
    FixedVarianceDecoder
from .iwae import IWAE
from .normalizing_autoencoder import NormalizingAutoEncoder
from .vae import VAE
from .vae_iaf import VAEIAF
from .nae_external import NaeExternal
import numpy as np

def get_model(model_name: str, decoder: str,
              latent_dims: int, img_shape: List, alpha: float, use_center_pixels: bool):
    model_dict = {'nae': NormalizingAutoEncoder,
                  'vae': VAE,
                  'iwae': IWAE,
                  'vae-iaf': VAEIAF,
                  'maf': MaskedAutoregressiveFlow,
                  'nae-ext': NaeExternal
                  }
    decoder_dict = {
        'fixed': FixedVarianceDecoder,
        'independent': IndependentVarianceDecoder,
        'dependent': LatentDependentDecoder,
    }

    vaes = ['nae', 'vae', 'iwae', 'vae-iaf', 'nae-ext']
    vae_channels = 64
    flow_features = 256

    decoder = decoder_dict[decoder](hidden_channels=vae_channels, output_shape=img_shape,
                                             latent_dim=latent_dims)
    encoder = ConvolutionalEncoder(hidden_channels=vae_channels, input_shape=img_shape, latent_dim=latent_dims)
    preprocessing_layers = [InverseTransform(AffineTransform(alpha, 1 - 2 * alpha)), Sigmoid(), ActNorm(img_shape[0])]
    if model_name in vaes:
        if model_name == 'nae':
            model = model_dict[model_name](encoder, decoder, preprocessing_layers, hardcoded_mask=use_center_pixels)
        elif model_name == 'nae-ext':
            model = model_dict[model_name](encoder, decoder, preprocessing_layers)
        else:
            model = model_dict[model_name](encoder, decoder)
    else:
        model = model_dict[model_name](np.prod(img_shape), hidden_features=flow_features, # TODO: dont hardcode
                                       num_layers=4, num_blocks_per_layer=2,
                                       preprocessing_layers=preprocessing_layers,
                                       act_norm_between_layers=True)
    return model
