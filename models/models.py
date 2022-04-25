from typing import List

import torch
from nflows.transforms import InverseTransform, AffineTransform

from bijectors.actnorm import ActNorm
from bijectors.sigmoid import Sigmoid
from flows.maf import MaskedAutoregressiveFlow
from models.autoencoder import IndependentVarianceDecoder, LatentDependentDecoder, ConvolutionalEncoder
from models.iwae import IWAE
from models.normalizing_autoencoder import NormalizingAutoEncoder
from models.vae import VAE
from models.vae_iaf import VAEIAF


def get_model(model_name: str, latent_dims: int, img_shape: List, alpha: float, use_center_pixels: bool,
              use_independent_variance_decoder=True):
    model_dict = {'nae': NormalizingAutoEncoder,
                  'vae': VAE,
                  'iwae': IWAE,
                  'vae-iaf': VAEIAF,
                  'maf': MaskedAutoregressiveFlow
                  }
    vaes = ['nae', 'vae', 'iwae', 'vae-iaf']
    vae_channels = 64
    flow_features = 256

    if use_independent_variance_decoder:
        decoder = IndependentVarianceDecoder(hidden_channels=vae_channels, output_shape=img_shape,
                                             latent_dim=latent_dims)
    else:
        decoder = LatentDependentDecoder(hidden_channels=vae_channels, output_shape=img_shape,
                                             latent_dim=latent_dims)
    encoder = ConvolutionalEncoder(hidden_channels=vae_channels, input_shape=img_shape, latent_dim=latent_dims)
    preprocessing_layers = [InverseTransform(AffineTransform(alpha, 1 - 2 * alpha)), Sigmoid(), ActNorm(img_shape[0])]
    if model_name in vaes:
        if model_name == 'nae':
            model = model_dict[model_name](encoder, decoder, preprocessing_layers, hardcoded_mask=use_center_pixels)
        else:
            model = model_dict[model_name](encoder, decoder)
    else:
        model = model_dict[model_name](784, hidden_features=flow_features, # TODO: dont hardcode
                                       num_layers=8, num_blocks_per_layer=2,
                                       preprocessing_layers=preprocessing_layers,
                                       act_norm_between_layers=True)
    return model
