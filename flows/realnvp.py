import torch
from nflows.nn import nets as nets
from nflows.transforms.base import CompositeTransform
from nflows.transforms.coupling import (
    AdditiveCouplingTransform,
    AffineCouplingTransform,
)
from torch.nn import functional as F

from flows.actnorm import ActNorm


def get_realnvp_bijector(features, hidden_features, num_layers,
                         num_blocks_per_layer,
                         use_volume_preserving=False,
                         activation=F.relu,
                         dropout_probability=0.0,
                         batch_norm_within_layers=False,
                         act_norm_between_layers=False, ):
    '''
    returns realnvp bijective transformation, only for flat tensors
    :return:
    '''

    if use_volume_preserving:
        coupling_constructor = AdditiveCouplingTransform
    else:
        coupling_constructor = AffineCouplingTransform

    mask = torch.ones(features)
    mask[::2] = -1

    def create_resnet(in_features, out_features):
        return nets.ResidualNet(
            in_features,
            out_features,
            hidden_features=hidden_features,
            num_blocks=num_blocks_per_layer,
            activation=activation,
            dropout_probability=dropout_probability,
            use_batch_norm=batch_norm_within_layers,
        )

    layers = []
    for _ in range(num_layers):
        transform = coupling_constructor(
            mask=mask, transform_net_create_fn=create_resnet
        )
        layers.append(transform)
        mask *= -1
        if act_norm_between_layers:
            layers.append(ActNorm(features=1))

    return CompositeTransform(layers)
