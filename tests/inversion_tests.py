from typing import List

import torch

from models.autoencoder import ConvolutionalEncoderSmall, IndependentVarianceDecoderSmall
from models.nae_internal import InternalLatentAutoEncoder


def test_inverse_nae(nae: InternalLatentAutoEncoder, img_dim: List, eps: float = 1e-6):
    input = torch.rand(4, *img_dim)
    z, deviations, _ = nae.embedding(input)
    returned_core, returned_shell = nae.forward(z, deviations)
    returned_input = nae.inverse_partition(returned_core, returned_shell)
    return torch.allclose(input, returned_input, atol=eps)


def test_forward_nae(nae: InternalLatentAutoEncoder, img_dim: List, eps: float = 1e-6):
    input = torch.rand(4, *img_dim)
    z, deviations = nae.partition(input)
    core, shell = nae.forward(z, deviations)
    output = nae.inverse_partition(core, shell)
    returned_z, returned_deviations, _ = nae.embedding(output)
    returned_input = nae.inverse_partition(returned_z, returned_deviations)
    return torch.allclose(input, returned_input, atol=eps)


num_channels = 3
core_size = 4
img_dim = [3, 28, 28]
encoder = ConvolutionalEncoderSmall(32, img_dim, 2)
decoder = IndependentVarianceDecoderSmall(32, img_dim, 2)
nae = InternalLatentAutoEncoder(encoder, decoder)
print(f"NAE (inverse -> forward) returns input: {test_inverse_nae(nae, img_dim)}")
print(f"NAE (forward -> inverse) returns input: {test_forward_nae(nae, img_dim)}")
