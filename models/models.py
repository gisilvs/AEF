from typing import List

import torch
from nflows.transforms import InverseTransform, AffineTransform

from bijectors.actnorm import ActNorm
from bijectors.masked_autoregressive_transform import get_masked_autoregressive_transform
from bijectors.sigmoid import Sigmoid
from flows.maf import MaskedAutoregressiveFlow
from .autoencoder import IndependentVarianceDecoder, LatentDependentDecoder, ConvolutionalEncoder, \
    FixedVarianceDecoder
from .iwae import IWAE
from .nae_internal import InternalLatentAutoEncoder
from .vae import VAE
from .vae_iaf import VAEIAF
from .nae_external import ExternalLatentAutoEncoder
import numpy as np
from bijectors.glow import SimpleGlow


def get_model(model_name: str, decoder: str,
              latent_dims: int, img_shape: List, alpha: float, use_glow=False):
    model_dict = {'nae-center': InternalLatentAutoEncoder,
                  'nae-corner': InternalLatentAutoEncoder,
                  'vae': VAE,
                  'iwae': IWAE,
                  'vae-iaf': VAEIAF,
                  'maf': MaskedAutoregressiveFlow,
                  'nae-external': ExternalLatentAutoEncoder
                  }
    decoder_dict = {
        'fixed': FixedVarianceDecoder,
        'independent': IndependentVarianceDecoder,
        'dependent': LatentDependentDecoder,
    }

    vaes = ['nae-center', 'nae-corner', 'nae-external', 'vae', 'iwae', 'vae-iaf', ]
    vae_channels = 64
    flow_features = 256

    decoder = decoder_dict[decoder](hidden_channels=vae_channels, output_shape=img_shape,
                                    latent_dim=latent_dims)
    encoder = ConvolutionalEncoder(hidden_channels=vae_channels, input_shape=img_shape, latent_dim=latent_dims)
    preprocessing_layers = [InverseTransform(AffineTransform(alpha, 1 - 2 * alpha)), Sigmoid(), ActNorm(img_shape[0])]
    if use_glow:
        preprocessing_layers.append(SimpleGlow(8, n_channels=img_shape[0], hidden_channels=256))
    if model_name in vaes:
        if 'nae' in model_name:
            # TODO: add core_flow selection for NAE
            core_flow_fn = get_masked_autoregressive_transform
            core_flow_pre = core_flow_fn(features=latent_dims,
                                         hidden_features=256,
                                         num_layers=4,
                                         num_blocks_per_layer=2,
                                         act_norm_between_layers=True)
            core_flow_post = core_flow_fn(features=latent_dims,
                                          hidden_features=256,
                                          num_layers=4,
                                          num_blocks_per_layer=2,
                                          act_norm_between_layers=True)
            if 'center' in model_name:
                model = model_dict[model_name](encoder=encoder, decoder=decoder, core_flow_pre=core_flow_pre,
                                               core_flow_post=core_flow_post, preprocessing_layers=preprocessing_layers,
                                               center_mask=True)
            elif 'corner' in model_name:
                model = model_dict[model_name](encoder=encoder, decoder=decoder, core_flow_pre=core_flow_pre,
                                               core_flow_post=core_flow_post, preprocessing_layers=preprocessing_layers,
                                               center_mask=False)
            elif 'ext' in model_name:
                model = model_dict[model_name](encoder=encoder, decoder=decoder, core_flow_pre=core_flow_pre,
                                               core_flow_post=core_flow_post, preprocessing_layers=preprocessing_layers)
        else:
            model = model_dict[model_name](encoder, decoder)
    else:
        model = model_dict[model_name](np.prod(img_shape), hidden_features=flow_features,  # TODO: dont hardcode
                                       num_layers=4, num_blocks_per_layer=2,
                                       preprocessing_layers=preprocessing_layers,
                                       act_norm_between_layers=True)
    return model
