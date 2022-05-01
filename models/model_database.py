from typing import List

from nflows.transforms import InverseTransform, AffineTransform

from bijectors.actnorm import ActNorm
from bijectors.masked_autoregressive_transform import get_masked_autoregressive_transform
from bijectors.sigmoid import Sigmoid
from flows.maf import MaskedAutoregressiveFlow
from .autoencoder import IndependentVarianceDecoderSmall, LatentDependentDecoderSmall, \
    FixedVarianceDecoderSmall, ConvolutionalEncoderSmall
from .iwae import IWAE
from .nae_internal import InternalLatentAutoEncoder
from .vae import VAE
from .vae_iaf import VAEIAF
from .nae_external import ExternalLatentAutoEncoder
import numpy as np

from .vdvae import FixedVarianceDecoderBig, IndependentVarianceDecoderBig, LatentDependentDecoderBig, \
    ConvolutionalEncoderBig


def get_model(model_name: str, architecture_size: str, decoder: str,
              latent_dims: int, img_shape: List, alpha: float):
    model_dict = {'nae-center': InternalLatentAutoEncoder,
                  'nae-corner': InternalLatentAutoEncoder,
                  'nae-external': ExternalLatentAutoEncoder,
                  'vae': VAE,
                  'iwae': IWAE,
                  'vae-iaf': VAEIAF,
                  'maf': MaskedAutoregressiveFlow,
                  }
    decoder_dict = {
        'fixed': {'small': FixedVarianceDecoderSmall, 'big': FixedVarianceDecoderBig},
        'independent': {'small': IndependentVarianceDecoderSmall, 'big': IndependentVarianceDecoderBig},
        'dependent': {'small': LatentDependentDecoderSmall, 'big': LatentDependentDecoderBig},
    }

    vaes = ['nae-center', 'nae-corner', 'nae-external', 'vae', 'iwae', 'vae-iaf', ]

    if architecture_size == 'small':
        vae_channels = 64
        decoder = decoder_dict[decoder]['small'](hidden_channels=vae_channels, output_shape=img_shape,
                                                 latent_ndims=latent_dims)
        encoder = ConvolutionalEncoderSmall(hidden_channels=vae_channels, input_shape=img_shape, latent_ndims=latent_dims)
    else:
        decoder = decoder_dict[decoder]['big'](output_shape=img_shape, latent_ndims=latent_dims)
        encoder = ConvolutionalEncoderBig(input_shape=img_shape, latent_ndims=latent_dims)

    preprocessing_layers = [InverseTransform(AffineTransform(alpha, 1 - 2 * alpha)), Sigmoid(), ActNorm(img_shape[0])]



    if model_name in vaes:
        if 'nae' in model_name:
            # TODO: add core_flow selection for NAE
            core_flow_fn = get_masked_autoregressive_transform
            if architecture_size == 'small':
                flow_features = 256
                num_layers = 4
            else:
                flow_features = 512
                num_layers = 8
            core_flow_pre = core_flow_fn(features=latent_dims,
                                         hidden_features=flow_features,
                                         num_layers=num_layers,
                                         num_blocks_per_layer=2,
                                         act_norm_between_layers=True)
            core_flow_post = core_flow_fn(features=latent_dims,
                                          hidden_features=flow_features,
                                          num_layers=num_layers,
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
        if architecture_size == 'small':
            flow_features = 512
            num_layers = 4
        else:
            flow_features = 1024
            num_layers = 16
        model = model_dict[model_name](np.prod(img_shape), hidden_features=flow_features,  # TODO: dont hardcode
                                       num_layers=num_layers, num_blocks_per_layer=2,
                                       preprocessing_layers=preprocessing_layers,
                                       act_norm_between_layers=True)
    return model
