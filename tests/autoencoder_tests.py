from typing import List

import torch

from models.autoencoder import IndependentVarianceDecoder, LatentDependentDecoder, ConvolutionalEncoder, \
    FixedVarianceDecoder
from models.autoencoder_base import AutoEncoder
from models.iwae import IWAE
from models.models import get_model
from models.nae_internal import InternalLatentAutoEncoder
from models.vae import VAE
from models.vae_iaf import VAEIAF
from models.nae_external import ExternalLatentAutoEncoder


def test_autoencoder_loss_backward(autoencoder: AutoEncoder, input_dims: List, n_iterations=10, batch_size=4):
    optimizer = torch.optim.Adam(params=autoencoder.parameters(), lr=1e-4)
    input_tensor = torch.rand(batch_size, *input_dims)

    for i in range(n_iterations):
        optimizer.zero_grad()
        loss = torch.mean(autoencoder.loss_function(input_tensor))
        loss.backward()
        optimizer.step()
    return True


def test_autoencoder_sample(autoencoder: AutoEncoder, n_samples=4):
    autoencoder.sample(n_samples)
    return True


def test_all_autoencoders(batch_size: int = 4, hidden_channels: int = 64):
    different_dims = [[1, 28, 28], [3, 32, 32]]
    latent_dims = [2, 4, 8, 16]
    decoder_names = ['fixed', 'independent', 'dependent']
    autoencoder_names = ['vae',
                         'iwae',
                         'vae-iaf',
                         'nae-center',
                         'nae-corner',
                         'nae-external'
                         ]
    # autoencoders = {
    #     # 'vae': VAE,
    #     # 'iwae': IWAE,
    #     'nae-center': NormalizingAutoEncoder,
    #     'nae-corner'
    #     # 'vae-iaf': VAEIAF,
    #     # 'nae-ext': NaeExternal,
    # }  # TODO: add NAE once it's refactored
    for input_dim in different_dims:
        for latent_dim in latent_dims:
            for decoder_name in decoder_names:
                for autoencoder_name in autoencoder_names:
                    # Add all tests here
                    ae = get_model(autoencoder_name, decoder_name, latent_dim, input_dim, 0.05)
                    test_autoencoder_loss_backward(ae, input_dim, n_iterations=10, batch_size=batch_size)
                    test_autoencoder_sample(ae)
    return True


def main():
    print(f"test_all_autoencoders: {'Success' if test_all_autoencoders() else 'Fail'}")


if __name__ == "__main__":
    main()
