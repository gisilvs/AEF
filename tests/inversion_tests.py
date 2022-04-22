import torch
from flows.realnvp import get_realnvp_bijector
from models.autoencoder import Encoder, LatentDependentDecoder
from models.normalizing_autoencoder import NormalizingAutoEncoder

def test_inverse_nae(nae: NormalizingAutoEncoder, num_channels=3, eps=1e-6):
    input = torch.rand(4, num_channels, 28, 28)
    z, deviations, _ = nae.embedding(input)
    returned_core, returned_shell = nae.forward(z, deviations)
    returned_input = nae.inverse_partition(returned_core, returned_shell)
    return torch.sum(torch.abs(input - returned_input) >= eps) == 0

def test_forward_nae(nae: NormalizingAutoEncoder, num_channels=3, eps=1e-6):
    input = torch.rand(4, num_channels, 28, 28)
    z, deviations = nae.partition(input)
    core, shell = nae.forward(z, deviations)
    output = nae.inverse_partition(core, shell)
    returned_z, returned_deviations, _ = nae.embedding(output)
    returned_input = nae.inverse_partition(returned_z, returned_deviations)
    return torch.sum(torch.abs(input - returned_input) >= eps) == 0

num_channels = 3
core_flow = get_realnvp_bijector(features=12, hidden_features=256, num_layers=6, num_blocks_per_layer=2, act_norm_between_layers=True)
encoder = Encoder(64,12, input_dim=[num_channels,28,28])
decoder = LatentDependentDecoder(64, 12, [num_channels,28,28])
mask = torch.zeros(28,28)
mask[13:15,13:15] = 1
nae = NormalizingAutoEncoder(core_flow, encoder, decoder, mask)
print(f"NAE (inverse -> forward) returns input: {test_inverse_nae(nae)}")
print(f"NAE (forward -> inverse) returns input: {test_forward_nae(nae)}")