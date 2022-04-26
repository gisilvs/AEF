import nflows.utils.typechecks as check
import torch
from nflows.transforms.base import Transform
from torch import nn


class ActNorm(Transform):
    def __init__(self, features):
        """
        Transform that performs activation normalization. Works for 2D and 4D inputs. For 4D
        inputs (images) normalization is performed per-channel, assuming BxCxHxW input shape.
        Reference:
        > D. Kingma et. al., Glow: Generative flow with invertible 1x1 convolutions, NeurIPS 2018.
        """
        if not check.is_positive_int(features):
            raise TypeError("Number of features must be a positive integer.")
        super().__init__()

        self.register_buffer("initialized", torch.tensor(False, dtype=torch.bool))
        self.log_scale = nn.Parameter(torch.zeros(features))
        self.shift = nn.Parameter(torch.zeros(features))

    @property
    def scale(self):
        return torch.exp(self.log_scale)

    def _broadcastable_scale_shift(self, inputs):
        if inputs.dim() == 4:
            return self.scale.view(1, -1, 1, 1), self.shift.view(1, -1, 1, 1)
        else:
            return self.scale.view(1, -1), self.shift.view(1, -1)

    def forward(self, inputs, context=None):
        if inputs.dim() not in [2, 4]:
            raise ValueError("Expecting inputs to be a 2D or a 4D tensor.")

        if self.training and not self.initialized:
            self._initialize_forward(inputs)

        scale, shift = self._broadcastable_scale_shift(inputs)
        outputs = (scale+1e-5) * inputs + shift

        if inputs.dim() == 4:
            batch_size, _, h, w = inputs.shape
            logabsdet = h * w * torch.sum(torch.log(scale + 1e-5)) * outputs.new_ones(batch_size)
        else:
            batch_size, w = inputs.shape
            logabsdet = w * torch.sum(torch.log(scale + 1e-5)) * outputs.new_ones(batch_size)

        return outputs, logabsdet

    def inverse(self, inputs, context=None):
        if inputs.dim() not in [2, 4]:
            raise ValueError("Expecting inputs to be a 2D or a 4D tensor.")

        if self.training and not self.initialized:
            self._initialize_inverse(inputs)

        scale, shift = self._broadcastable_scale_shift(inputs)
        outputs = (inputs - shift) / (scale+1e-5)

        if inputs.dim() == 4:
            batch_size, _, h, w = inputs.shape
            logabsdet = -h * w * torch.sum(torch.log(scale + 1e-5)) * outputs.new_ones(batch_size)
        else:
            batch_size, w = inputs.shape
            logabsdet = - w * torch.sum(torch.log(scale + 1e-5)) * outputs.new_ones(batch_size)

        return outputs, logabsdet

    def _initialize_forward(self, inputs):
        """Data-dependent initialization, s.t. post-actnorm activations have zero mean and unit
        variance. """
        if inputs.dim() == 4:
            num_channels = inputs.shape[1]
            inputs = inputs.permute(0, 2, 3, 1).reshape(-1, num_channels)
        else:
            inputs = inputs.view(-1,1)

        with torch.no_grad():
            std = inputs.std(dim=0)
            mu = (inputs / std).mean(dim=0)
            self.log_scale.data = -torch.log(std)
            self.shift.data = -mu
            self.initialized.data = torch.tensor(True, dtype=torch.bool)

    def _initialize_inverse(self, inputs):
        """Data-dependent initialization, s.t. post-actnorm activations have zero mean and unit
        variance. """
        if inputs.dim() == 4:
            num_channels = inputs.shape[1]
            inputs = inputs.permute(0, 2, 3, 1).reshape(-1, num_channels)
        else:
            inputs = inputs.view(-1,1)

        with torch.no_grad():
            mu = inputs.mean(dim=0)
            std = (inputs - mu).std(dim=0)
            self.log_scale.data = torch.log(std)
            self.shift.data = mu
            self.initialized.data = torch.tensor(True, dtype=torch.bool)


