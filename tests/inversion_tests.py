import torch
from flows.realnvp import get_realnvp_bijector
from models.autoencoder import Encoder, LatentDependentDecoder
from models.normalizing_autoencoder import NormalizingAutoEncoder

def test_inverse_nae(nae: NormalizingAutoEncoder, num_channels=3, eps=1e-6):
    input = torch.rand(4, num_channels, 28, 28)
    z, deviations, _ = nae.embedding(input)
    returned_core, returned_shell = nae.forward(z, deviations)
    returned_input = nae.inverse_partition(returned_core, returned_shell)
    return torch.allclose(input, returned_input, atol=1e-6)

def test_forward_nae(nae: NormalizingAutoEncoder, num_channels=3, eps=1e-6):
    input = torch.rand(4, num_channels, 28, 28)
    z, deviations = nae.partition(input)
    core, shell = nae.forward(z, deviations)
    output = nae.inverse_partition(core, shell)
    returned_z, returned_deviations, _ = nae.embedding(output)
    returned_input = nae.inverse_partition(returned_z, returned_deviations)
    return torch.allclose(input, returned_input, atol=1e-6)

num_channels = 3
core_size = 4
nae = NormalizingAutoEncoder(core_size, [3,28,28])
print(f"NAE (inverse -> forward) returns input: {test_inverse_nae(nae)}")
print(f"NAE (forward -> inverse) returns input: {test_forward_nae(nae)}")