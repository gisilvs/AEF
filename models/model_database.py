from typing import List

from nflows.transforms import InverseTransform, AffineTransform, IdentityTransform

from bijectors.actnorm import ActNorm
from bijectors.masked_autoregressive_transform import get_masked_autoregressive_transform
from bijectors.sigmoid import Sigmoid
from flows.maf import MaskedAutoregressiveFlow
from .autoencoder import IndependentVarianceDecoderSmall, LatentDependentDecoderSmall, \
    FixedVarianceDecoderSmall, ConvolutionalEncoderSmall
from .denoising_ae import DeterministicConvolutionalEncoderSmall, DeterministicConvolutionalDecoderSmall, \
    DenoisingAutoEncoder
from .iwae import IWAE, ExtendedIWAE, DenoisingIWAE
from .nae_internal import InternalLatentAutoEncoder

from .vae import VAE, ExtendedVAE, DenoisingVAE, ExtendedDenoisingVAE
from .vae_iaf import VAEIAF, ExtendedVAEIAF, DenoisingVAEIAF
from .nae_external import ExternalLatentAutoEncoder
import numpy as np

from .vdvae import FixedVarianceDecoderBig, IndependentVarianceDecoderBig, LatentDependentDecoderBig, \
    ConvolutionalEncoderBig


def get_model(model_name: str, architecture_size: str, decoder: str,
              latent_dims: int, img_shape: List, alpha: float,
              posterior_flow_name: str, prior_flow_name: str, test=False):

    decoder_dict = {
        'fixed': {'small': FixedVarianceDecoderSmall, 'big': FixedVarianceDecoderBig},
        'independent': {'small': IndependentVarianceDecoderSmall, 'big': IndependentVarianceDecoderBig},
        'dependent': {'small': LatentDependentDecoderSmall, 'big': LatentDependentDecoderBig},
    }

    model_dict = {
        'nae-center': {'default': InternalLatentAutoEncoder, 'extended': InternalLatentAutoEncoder},
        'nae-corner': {'default': InternalLatentAutoEncoder, 'extended': InternalLatentAutoEncoder},
        'nae-external': {'default': ExternalLatentAutoEncoder, 'extended': ExternalLatentAutoEncoder},
        'vae': {'default': VAE, 'extended': ExtendedVAE},
        'iwae': {'default': IWAE, 'extended': ExtendedIWAE},
        'vae-iaf': {'default': VAEIAF, 'extended': VAEIAF}  # TODO: decide what to do here...
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

    if 'nae' in model_name:
        core_flow_pre, core_flow_post = get_flows(latent_dims, model_name, posterior_flow_name, prior_flow_name,
                                                  architecture_size)
        if model_name == 'nae-center':
            model = model_dict[model_name]['default'](encoder=encoder, decoder=decoder, core_flow_pre=core_flow_pre,
                                                      core_flow_post=core_flow_post,
                                                      preprocessing_layers=preprocessing_layers,
                                                      center_mask=True)
        elif model_name == 'nae-corner':
            model = model_dict[model_name]['default'](encoder=encoder, decoder=decoder, core_flow_pre=core_flow_pre,
                                                      core_flow_post=core_flow_post,
                                                      preprocessing_layers=preprocessing_layers,
                                                      center_mask=False)
        elif model_name == 'nae-external':
            model = model_dict[model_name]['default'](encoder=encoder, decoder=decoder, core_flow_pre=core_flow_pre,
                                                      core_flow_post=core_flow_post,
                                                      preprocessing_layers=preprocessing_layers)

    elif (posterior_flow_name == 'none') and (prior_flow_name == 'none'):
        model = model_dict[model_name]['default'](encoder=encoder, decoder=decoder)
    else:
        posterior_flow, prior_flow = get_flows(latent_dims, model_name, posterior_flow_name, prior_flow_name,
                                               architecture_size)

        model = model_dict[model_name]['extended'](encoder=encoder, decoder=decoder, posterior_bijector=posterior_flow,
                                           prior_bijector=prior_flow)
    return model


def get_flows(latent_dims, model_name, posterior_flow_name, prior_flow_name, architecture_size):
    # TODO: check if correct
    if architecture_size == 'big':
        flow_features = 512
        num_layers = 8
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
        if 'nae' in model_name:
            post_flow = get_masked_autoregressive_transform(features=latent_dims,
                                                            hidden_features=flow_features,
                                                            num_layers=num_layers,
                                                            num_blocks_per_layer=2,
                                                            act_norm_between_layers=True)
        else:
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
        if 'nae' in model_name:
            prior_flow = get_masked_autoregressive_transform(features=latent_dims,
                                                             hidden_features=flow_features,
                                                             num_layers=num_layers,
                                                             num_blocks_per_layer=2,
                                                             act_norm_between_layers=True)
        else:
            prior_flow = IdentityTransform()
    else:
        raise ValueError

    return post_flow, prior_flow


def get_model_denoising(model_name: str, decoder: str, latent_dims: int, img_shape: List, alpha: float):

    decoder_dict = {
        'fixed': FixedVarianceDecoderSmall,
        'independent': IndependentVarianceDecoderSmall,
        'dependent': LatentDependentDecoderSmall
    }

    preprocessing_layers = [InverseTransform(AffineTransform(alpha, 1 - 2 * alpha)), Sigmoid(),
                            ActNorm(img_shape[0])]

    if model_name == 'vae-iaf-maf':
        vae_channels = 64
        encoder = DeterministicConvolutionalEncoderSmall(vae_channels, input_shape=img_shape, latent_dims=latent_dims)
        decoder = FixedVarianceDecoderSmall(vae_channels, output_shape=img_shape, latent_dims=latent_dims)

        flow_features = 256
        num_layers = 4
        prior_flow = get_masked_autoregressive_transform(features=latent_dims,
                                                         hidden_features=flow_features,
                                                         num_layers=num_layers,
                                                         num_blocks_per_layer=2,
                                                         act_norm_between_layers=True)
        post_flow = get_masked_autoregressive_transform(features=latent_dims,
                                                        hidden_features=flow_features,
                                                        num_layers=num_layers,
                                                        num_blocks_per_layer=2,
                                                        act_norm_between_layers=True,
                                                        is_inverse=True)

        model = ExtendedDenoisingVAE(encoder, decoder, post_flow, prior_flow)
        return model

    if model_name == 'ae':
        vae_channels = 64
        encoder = DeterministicConvolutionalEncoderSmall(vae_channels, input_shape=img_shape, latent_dims=latent_dims)
        decoder = DeterministicConvolutionalDecoderSmall(vae_channels, output_shape=img_shape, latent_dims=latent_dims)
        model = DenoisingAutoEncoder(encoder, decoder)
        return model
    elif model_name == 'maf':


        flow_features = 256
        num_layers = 4

        model = MaskedAutoregressiveFlow(int(np.prod(img_shape)), hidden_features=flow_features,  # TODO: dont hardcode
                                       num_layers=num_layers, num_blocks_per_layer=2,
                                       preprocessing_layers=preprocessing_layers,
                                       act_norm_between_layers=True)
        return model


    vae_channels = 64
    decoder = decoder_dict[decoder](hidden_channels=vae_channels, output_shape=img_shape,
                                             latent_dims=latent_dims)
    encoder = ConvolutionalEncoderSmall(hidden_channels=vae_channels, input_shape=img_shape,
                                        latent_dims=latent_dims)

    model_dict = {'nae-center': InternalLatentAutoEncoder,
                  'nae-corner': InternalLatentAutoEncoder,
                  'nae-external': ExternalLatentAutoEncoder,
                  'vae': DenoisingVAE,
                  'iwae': DenoisingIWAE,
                  'vae-iaf': DenoisingVAEIAF,
                  }

    core_flow_pre = get_masked_autoregressive_transform(features=latent_dims,
                                                        hidden_features=256,
                                                        num_layers=4,
                                                        num_blocks_per_layer=2,
                                                        act_norm_between_layers=True)
    core_flow_post = get_masked_autoregressive_transform(features=latent_dims,
                                                         hidden_features=256,
                                                         num_layers=4,
                                                         num_blocks_per_layer=2,
                                                         act_norm_between_layers=True)
    if model_name == 'nae-center':
        model = model_dict[model_name](encoder=encoder, decoder=decoder, core_flow_pre=core_flow_pre,
                                       core_flow_post=core_flow_post, preprocessing_layers=preprocessing_layers,
                                       center_mask=True)
    elif model_name == 'nae-corner':
        model = model_dict[model_name](encoder=encoder, decoder=decoder, core_flow_pre=core_flow_pre,
                                       core_flow_post=core_flow_post, preprocessing_layers=preprocessing_layers,
                                       center_mask=False)
    elif model_name == 'nae-external':
        model = model_dict[model_name](encoder=encoder, decoder=decoder, core_flow_pre=core_flow_pre,
                                       core_flow_post=core_flow_post, preprocessing_layers=preprocessing_layers)
    else:
        model = model_dict[model_name](encoder=encoder, decoder=decoder)
    return model