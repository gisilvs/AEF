import torch
from models.autoencoder import IndependentVarianceDecoder, LatentDependentDecoder, Encoder

from datasets import get_train_val_dataloaders
from models.autoencoder_base import AutoEncoder
from typing import List

from models.iwae import IWAE
from models.vae import VAE


def test_autoencoder_loss_backward(autoencoder: AutoEncoder, input_dims: List, n_iterations=100, batch_size=4):
    optimizer = torch.optim.SGD(params=autoencoder.parameters(), lr=1e-4)
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
    decoders = [IndependentVarianceDecoder, LatentDependentDecoder]
    autoencoders = [VAE, IWAE]  # TODO: add NAE once it's refactored
    for input_dim in different_dims:
        for latent_dim in latent_dims:
            #encoder = Encoder(hidden_channels, latent_dim, input_dim)
            for decoder_class in decoders:
                for autoencoder_class in autoencoders:
                    # Add all tests here
                    #decoder = decoder_class(hidden_channels, latent_dim, input_dim)
                    ae = autoencoder_class(hidden_channels, latent_dim, input_dim)
                    test_autoencoder_loss_backward(ae, input_dim, n_iterations=10, batch_size=batch_size)
                    test_autoencoder_sample(ae)
    return True

def main():
    print(f"test_all_autoencoders: {'Success' if test_all_autoencoders() else 'Fail'}")

if __name__ == "__main__":
    main()
