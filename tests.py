from main import NormalizingAutoEncoder, RealNVP, Encoder, Decoder
import torch

def test_inverse_nae(nae: NormalizingAutoEncoder, num_channels=3, eps=1e-6):
    input = torch.rand(4, num_channels, 28, 28)
    core, shell = nae.partition(input)
    z, deviations, _ = nae.embedding(core, shell)
    returned_core, returned_shell = nae.forward(z, deviations)
    returned_input = nae.inverse_partition(returned_core, returned_shell)
    return torch.sum(torch.abs(input - returned_input) >= eps) == 0

def test_forward_nae(nae: NormalizingAutoEncoder, num_channels=3, eps=1e-6):
    input = torch.rand(4, num_channels, 28, 28)
    z, deviations = nae.partition(input)
    core, shell = nae.forward(z, deviations)
    returned_z, returned_deviations, _ = nae.embedding(core, shell)
    returned_input = nae.inverse_partition(returned_z, returned_deviations)
    return torch.sum(torch.abs(input - returned_input) >= eps) == 0

def test_inverse_flow(flow, eps=1e-6):
    input = torch.rand(4, flow.input_dim)
    output = flow.forward(flow.inverse(input)[0])
    return torch.sum(torch.abs(input - output) >= eps) == 0

num_channels = 3
core_flow = RealNVP(input_dim=12, num_flows=6, hidden_units=256)
encoder = Encoder(64,12, input_channels=3)
decoder = Decoder(64, 12, [num_channels,28,28])
mask = torch.zeros(28,28)
mask[13:15,13:15] = 1
nae = NormalizingAutoEncoder(core_flow, encoder, decoder, mask)
print(f"NAE (inverse -> forward) returns input: {test_inverse_nae(nae)}")
print(f"NAE (forward -> inverse) returns input: {test_forward_nae(nae)}")
print(f"RealNVP returns input: {test_inverse_flow(core_flow)}")