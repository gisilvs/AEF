from typing import List

import torch

from models.autoencoder_base import AutoEncoder
from models.model_database import get_model


def test_autoencoder_loss_backward(autoencoder: AutoEncoder, input_dims: List, n_iterations=10, batch_size=4,
                                   device=torch.device("cpu")):
    optimizer = torch.optim.Adam(params=autoencoder.parameters(), lr=1e-4)
    input_tensor = torch.rand(batch_size, *input_dims).to(device)

    for i in range(n_iterations):
        optimizer.zero_grad()
        loss = torch.mean(autoencoder.loss_function(input_tensor))
        loss.backward()
        optimizer.step()
    return True


def test_autoencoder_sample(autoencoder: AutoEncoder, n_samples=4):
    autoencoder.sample(n_samples)
    return True


def test_autoencoders_without_flows(batch_size: int = 4):
    different_dims = [[1, 28, 28], [3, 32, 32]]
    latent_dims = [2, 8, 32]#[2, 4, 8, 16, 32]
    decoder_names = ['dependent', 'fixed', 'independent', ]
    architecture_sizes = ['big', 'small']
    autoencoder_names = ['vae',
                         'iwae',
                         'vae-iaf',
                         'aef-center',
                         'aef-corner',
                         'aef-linear'
                         ]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")
    for architecture_size in architecture_sizes:
        for input_dim in different_dims:
            if architecture_size == 'big' and input_dim == [1, 28, 28]:
                continue
            for latent_dim in latent_dims:
                for decoder_name in decoder_names:
                    for autoencoder_name in autoencoder_names:
                        # Add all tests here
                        ae = get_model(autoencoder_name, architecture_size, decoder_name, latent_dim, input_dim,
                                       0.05, 'none', 'none', test=True).to(device)
                        test_autoencoder_loss_backward(ae, input_dim, n_iterations=10, batch_size=batch_size, device=device)
                        test_autoencoder_sample(ae)
                        print(f'Tested {autoencoder_name} {decoder_name} {architecture_size} {latent_dim} {input_dim}')
                        del ae
                        torch.cuda.empty_cache()
    return True

def test_autoencoders_with_flows(batch_size: int = 4):
    different_dims = [[1, 28, 28], [3, 32, 32]]
    latent_dims = [2]
    decoder_names = ['fixed']
    architecture_sizes = ['small']
    autoencoder_names = ['vae',
                         'iwae',
                         'vae-iaf',
                         'aef-center',
                         'aef-corner',
                         'aef-linear'
                         ]
    flow_names = ['maf', 'iaf', 'none']

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")
    for architecture_size in architecture_sizes:
        for input_dim in different_dims:
            if architecture_size == 'big' and input_dim == [1, 28, 28]:
                continue
            for latent_dim in latent_dims:
                for decoder_name in decoder_names:
                    for autoencoder_name in autoencoder_names:
                        for posterior_flow in flow_names:
                            for prior_flow in flow_names:
                                # Add all tests here
                                ae = get_model(autoencoder_name, architecture_size, decoder_name, latent_dim, input_dim,
                                               0.05, posterior_flow, prior_flow, test=True).to(device)
                                test_autoencoder_loss_backward(ae, input_dim, n_iterations=10, batch_size=batch_size, device=device)
                                test_autoencoder_sample(ae)
                                print(f'Tested {autoencoder_name} {decoder_name} {architecture_size} {posterior_flow} {prior_flow} {latent_dim} {input_dim}')
                                del ae
                                torch.cuda.empty_cache()
    return True


def main():
    print(f"test_autoencoders_without_flows: {'Success' if test_autoencoders_without_flows() else 'Fail'}")
    print(f"test_autoencoders_with_flows: {'Success' if test_autoencoders_without_flows() else 'Fail'}")


if __name__ == "__main__":
    main()
