import torch
from nflows.transforms import Transform, InputOutsideDomain
from nflows.utils import torchutils
import torch.nn.functional as F


class Sigmoid(Transform):
    def __init__(self, eps=1e-6):
        super(Sigmoid, self).__init__()
        eps = torch.Tensor([eps])
        self.register_buffer('eps', eps)

    def forward(self, inputs, context=None):
        outputs = torch.sigmoid(inputs)
        logabsdet = torchutils.sum_except_batch(
            - F.softplus(-inputs) - F.softplus(inputs)
        )
        return outputs, logabsdet

    def inverse(self, inputs, context=None):
        if torch.min(inputs) < 0 or torch.max(inputs) > 1:
            raise InputOutsideDomain()

        inputs = torch.clamp(inputs, self.eps, 1 - self.eps)

        outputs = torch.log(inputs) - torch.log1p(-inputs)
        logabsdet = -torchutils.sum_except_batch(
            - F.softplus(-outputs)
            - F.softplus(outputs)
        )
        return outputs, logabsdet
