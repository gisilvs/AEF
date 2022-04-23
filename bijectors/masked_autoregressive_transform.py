"""Implementations of autoregressive bijectors."""

from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
from nflows.transforms.base import CompositeTransform
from nflows.transforms.base import InverseTransform
from nflows.transforms.permutations import RandomPermutation, ReversePermutation
from torch.nn import functional as F

from bijectors.actnorm import ActNorm


# todo: check which is the right direction for maf and iaf
def get_autoregressive_flow(
        features,
        hidden_features,
        num_layers,
        num_blocks_per_layer,
        use_residual_blocks=True,
        use_random_masks=False,
        use_random_permutations=False,
        activation=F.relu,
        dropout_probability=0.0,
        batch_norm_within_layers=False,
        act_norm_between_layers=False,
        is_inverse=False):
    if use_random_permutations:
        permutation_constructor = RandomPermutation
    else:
        permutation_constructor = ReversePermutation

    layers = []
    for _ in range(num_layers):
        layers.append(permutation_constructor(features))
        layers.append(
            MaskedAffineAutoregressiveTransform(
                features=features,
                hidden_features=hidden_features,
                num_blocks=num_blocks_per_layer,
                use_residual_blocks=use_residual_blocks,
                random_mask=use_random_masks,
                activation=activation,
                dropout_probability=dropout_probability,
                use_batch_norm=batch_norm_within_layers,
            )
        )
        if act_norm_between_layers:
            if is_inverse:
                layers.append(ActNorm(features=1))
            else:
                layers.append(InverseTransform(ActNorm(features=1)))

    if is_inverse:
        return CompositeTransform(layers)
    else:
        return InverseTransform(CompositeTransform(layers))
