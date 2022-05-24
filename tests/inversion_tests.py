from typing import List

import torch
from nflows.transforms import InverseTransform, AffineTransform

from bijectors.actnorm import ActNorm
from bijectors.masked_autoregressive_transform import get_masked_autoregressive_transform
from bijectors.sigmoid import Sigmoid
from models.autoencoder import ConvolutionalEncoderSmall, IndependentVarianceDecoderSmall
from models.aef_internal import InternalAEF

import util
from models.aef_linear import LinearAEF


def test_inverse_internal_aef(aef: InternalAEF, img_dim: List, eps: float = 1e-6):
    with torch.no_grad():
        x = torch.rand(4, *img_dim)
        z_1, deviations, _ = aef.embedding(x)
        returned_x = aef.forward(z_1, deviations)
        return torch.allclose(x, returned_x, atol=eps)


def test_forward_internal_aef(aef: InternalAEF, img_dim: List, eps: float = 1e-6):
    with torch.no_grad():
        input = torch.randn(4, *img_dim)
        z_1, deviations = aef.partition(input)
        output = aef.forward(z_1, deviations)
        returned_z, returned_deviations, _ = aef.embedding(output)
        returned_input = aef.inverse_partition(returned_z, returned_deviations)
        return torch.allclose(input, returned_input, atol=eps)

def test_inverse_linear_aef(aef: LinearAEF, img_dim: List, eps: float = 1e-6):
    with torch.no_grad():
        input = torch.rand(4, *img_dim)
        z, deviations, _ = aef.embedding(input)
        returned_input = aef.forward(z, deviations)
        return torch.allclose(input, returned_input, atol=eps)



if __name__ == "__main__":
    hidden_channels = 32
    core_size = 4
    img_dim = [3, 32, 32]
    encoder = ConvolutionalEncoderSmall(hidden_channels, img_dim, core_size)
    decoder = IndependentVarianceDecoderSmall(hidden_channels, img_dim, core_size)

    flow_features = 32
    num_layers = 2
    core_encoder = get_masked_autoregressive_transform(features=core_size, hidden_features=flow_features,
                                                       num_layers=num_layers, num_blocks_per_layer=2,
                                                       act_norm_between_layers=True)
    prior_flow = get_masked_autoregressive_transform(features=core_size, hidden_features=flow_features,
                                                     num_layers=num_layers, num_blocks_per_layer=2,
                                                     act_norm_between_layers=True)
    mask = util.get_center_mask(img_dim, core_size)
    aef_center = InternalAEF(encoder, decoder, core_encoder, prior_flow, mask )
    print(f"AEF-center w/o preprocessing (inverse -> forward) returns input: {test_inverse_internal_aef(aef_center, img_dim)}")
    print(f"AEF-center w/o preprocessing (forward -> inverse) returns input: {test_forward_internal_aef(aef_center, img_dim)}")

    aef_linear = LinearAEF(encoder, decoder, core_encoder, prior_flow)
    print(
        f"AEF-linear w/o preprocessing (inverse -> forward) returns input: {test_inverse_linear_aef(aef_linear, img_dim)}")
    print(
        f"AEF-linear w/o preprocessing (forward -> inverse) returns input: {test_forward_linear_aef(aef_linear, core_size, img_dim)}")

    alpha = 1e-6
    preprocessing_layers = [InverseTransform(AffineTransform(alpha, 1 - 2 * alpha)), Sigmoid(),
                            ActNorm(img_dim[0])]
    aef_center = InternalAEF(encoder, decoder, core_encoder, prior_flow, mask, preprocessing_layers=preprocessing_layers)
    print(f"AEF-center w/ preprocessing (inverse -> forward) returns input: {test_inverse_internal_aef(aef_center, img_dim)}")
    print(f"AEF-center w/ preprocessing (forward -> inverse) returns input: {test_forward_internal_aef(aef_center, img_dim)}")

    aef_linear = LinearAEF(encoder, decoder, core_encoder, prior_flow,
                           preprocessing_layers=preprocessing_layers)
    print(
        f"AEF-linear w/ preprocessing (inverse -> forward) returns input: {test_inverse_linear_aef(aef_linear, img_dim)}")
    print(
        f"AEF-linear w/ preprocessing (forward -> inverse) returns input: {test_forward_linear_aef(aef_linear, core_size, img_dim)}")

    exit()
