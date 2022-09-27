from typing import List

from nflows.transforms import InverseTransform, AffineTransform, IdentityTransform

import util
from bijectors.actnorm import ActNorm
from bijectors.masked_autoregressive_transform import get_masked_autoregressive_transform
from bijectors.sigmoid import Sigmoid
from flows.maf import MaskedAutoregressiveFlow
from .autoencoder import LearnableVarianceDecoderSmall, LatentDependentDecoderSmall, \
    FixedVarianceDecoderSmall, ConvolutionalEncoderSmall
from .denoising_ae import DeterministicConvolutionalEncoderSmall, DeterministicConvolutionalDecoderSmall, \
    StandardAutoEncoder, DeterministicConvolutionalDecoderBig, DeterministicConvolutionalEncoderBig
from .iwae import ExtendedIWAE
from .aef_internal import InternalAEF


from .vae import ExtendedVAE
from .aef_linear import LinearAEF
import numpy as np

from .vdvae import FixedVarianceDecoderBig, LearnableVarianceDecoderBig, LatentDependentDecoderBig, \
    ConvolutionalEncoderBig


def get_model(model_name: str, architecture_size: str, decoder: str,
              latent_dims: int, img_shape: List, alpha: float,
              posterior_flow_name: str, prior_flow_name: str, test=False):

    decoder_dict = {
        'fixed': {'small': FixedVarianceDecoderSmall, 'big': FixedVarianceDecoderBig},
        'independent': {'small': LearnableVarianceDecoderSmall, 'big': LearnableVarianceDecoderBig},
        'dependent': {'small': LatentDependentDecoderSmall, 'big': LatentDependentDecoderBig},
    }

    model_dict = {
        'aef-center': InternalAEF,
        'aef-corner': InternalAEF,
        'aef-linear': LinearAEF,
        'vae': ExtendedVAE,
        'iwae': ExtendedIWAE,
    }

    preprocessing_layers = [InverseTransform(AffineTransform(alpha, 1 - 2 * alpha)), Sigmoid(),
                            ActNorm(img_shape[0])]

    if model_name == 'maf':
        if architecture_size == 'small':
            flow_features = 256
            num_layers = 4
        else:
            flow_features = 512
            num_layers = 8
        model = MaskedAutoregressiveFlow(int(np.prod(img_shape)), hidden_features=flow_features,  # TODO: dont hardcode
                                       num_layers=num_layers, num_blocks_per_layer=2,
                                       preprocessing_layers=preprocessing_layers,
                                       act_norm_between_layers=True)
        return model
    if architecture_size == 'small':
        vae_channels = 64
        decoder = decoder_dict[decoder]['small'](hidden_channels=vae_channels, output_shape=img_shape,
                                                 latent_dims=latent_dims)
        encoder = ConvolutionalEncoderSmall(hidden_channels=vae_channels, input_shape=img_shape,
                                            latent_dims=latent_dims)
    else:
        if test:
            decoder = decoder_dict[decoder]['big'](output_shape=img_shape, latent_dims=latent_dims, size='test')
            encoder = ConvolutionalEncoderBig(input_shape=img_shape, latent_dims=latent_dims, size='test')
        else:
            decoder = decoder_dict[decoder]['big'](output_shape=img_shape, latent_dims=latent_dims)
            encoder = ConvolutionalEncoderBig(input_shape=img_shape, latent_dims=latent_dims)

    if model_name == 'ae':
        if architecture_size == 'small':
            vae_channels = 64
            encoder = DeterministicConvolutionalEncoderSmall(vae_channels, img_shape, latent_dims)
            decoder = DeterministicConvolutionalDecoderSmall(vae_channels, img_shape, latent_dims)
        else:
            encoder = DeterministicConvolutionalEncoderBig(img_shape, latent_dims)
            decoder = DeterministicConvolutionalDecoderBig(img_shape, latent_dims)

        return StandardAutoEncoder(encoder, decoder, preprocessing_layers)
    if 'aef' in model_name:
        core_encoder, prior_flow = get_flows(latent_dims, posterior_flow_name, prior_flow_name,
                                                  architecture_size)
        if model_name == 'aef-center':
            mask = util.get_center_mask(img_shape, latent_dims)
            model = model_dict[model_name](encoder=encoder, decoder=decoder, core_encoder=core_encoder,
                                                      prior_flow=prior_flow, mask=mask,
                                                      preprocessing_layers=preprocessing_layers)
        elif model_name == 'aef-corner':
            mask = util.get_corner_mask(img_shape, latent_dims)
            model = model_dict[model_name](encoder=encoder, decoder=decoder, core_encoder=core_encoder,
                                                      prior_flow=prior_flow, mask=mask,
                                                      preprocessing_layers=preprocessing_layers)
        elif model_name == 'aef-linear':
            model = model_dict[model_name](encoder=encoder, decoder=decoder, core_encoder=core_encoder,
                                                      prior_flow=prior_flow,
                                                      preprocessing_layers=preprocessing_layers)

    else:
        posterior_flow, prior_flow = get_flows(latent_dims, posterior_flow_name, prior_flow_name,
                                               architecture_size)

        model = model_dict[model_name](encoder=encoder, decoder=decoder, posterior_bijector=posterior_flow,
                                           prior_bijector=prior_flow, preprocessing_layers=preprocessing_layers)
    return model


def get_flows(latent_dims, posterior_flow_name, prior_flow_name, architecture_size):
    if architecture_size == 'big':
        flow_features = 256
        num_layers = 4
    elif architecture_size == 'small':
        flow_features = 256
        num_layers = 4
    else:
        raise ValueError

    if posterior_flow_name == 'maf':
        post_flow = get_masked_autoregressive_transform(features=latent_dims,
                                                        hidden_features=flow_features,
                                                        num_layers=num_layers,
                                                        num_blocks_per_layer=2,
                                                        act_norm_between_layers=True)
    elif posterior_flow_name == 'iaf':
        post_flow = get_masked_autoregressive_transform(features=latent_dims,
                                                        hidden_features=flow_features,
                                                        num_layers=num_layers,
                                                        num_blocks_per_layer=2,
                                                        act_norm_between_layers=True,
                                                        is_inverse=True)
    elif posterior_flow_name == 'none':
        post_flow = IdentityTransform()
    else:
        raise ValueError
    if prior_flow_name == 'maf':
        prior_flow = get_masked_autoregressive_transform(features=latent_dims,
                                                         hidden_features=flow_features,
                                                         num_layers=num_layers,
                                                         num_blocks_per_layer=2,
                                                         act_norm_between_layers=True)
    elif prior_flow_name == 'iaf':
        prior_flow = get_masked_autoregressive_transform(features=latent_dims,
                                                         hidden_features=flow_features,
                                                         num_layers=num_layers,
                                                         num_blocks_per_layer=2,
                                                         act_norm_between_layers=True,
                                                         is_inverse=True)
    elif prior_flow_name == 'none':
        prior_flow = IdentityTransform()
    else:
        raise ValueError

    return post_flow, prior_flow


