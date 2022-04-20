import torch
import torch.nn as nn
import normflow as nf
import numpy as np

class RealNVP(nn.Module):
    def __init__(self, input_dim: int, num_flows: int, hidden_units: int = None):
        super().__init__()

        self.input_dim = input_dim
        self.num_flows = num_flows
        if hidden_units is None:
            hidden_units = 2 * input_dim

        masks = torch.zeros(num_flows, input_dim)
        masks[::2, ::2] = 1
        masks[1::2, 1::2] = 1
        self.register_buffer('masks', masks)  # Ensure masks are transferred to device

        flows = []
        for i in range(num_flows):
            s = nf.nets.MLP([input_dim, hidden_units, input_dim], init_zeros=True)
            t = nf.nets.MLP([input_dim, hidden_units, input_dim], init_zeros=True)
            flows += [nf.flows.MaskedAffineFlow(masks[i, :], t, s)]
            #flows += [nf.flows.ActNorm(input_dim)]
        self.flows = nn.ModuleList(flows)

    def forward(self, z):
        log_det = torch.zeros(z.shape[0], device=z.device)
        for flow in self.flows:
            z, log_det_z = flow(z)
            log_det += log_det_z
        x = z
        return x#, log_det # TODO: decide whether we want to return the log_determinant in the forward direction

    def inverse(self, x):
        log_det = torch.zeros(x.shape[0], device=x.device)
        z = x
        for i in range(len(self.flows) - 1, -1, -1):
            z, log_det_z = self.flows[i].inverse(z)
            log_det += log_det_z
        return z, log_det

