import matplotlib.pyplot as plt

from util import get_avg_loss_over_iterations, plot_loss_over_iterations
import normflow as nf
import torch

import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from tqdm import tqdm
from flows.realnvp import RealNVP
from models.autoencoder import Encoder, Decoder
from models.normalizing_autoencoder import NormalizingAutoEncoder
from util import make_averager, refresh_bar, plot_loss, dequantize

def main():
    torch_generator = torch.Generator().manual_seed(3)

    # 2-d latent space, parameter count in same order of magnitude
    # as in the original VAE paper (VAE paper has about 3x as many)
    n_iterations=1000
    latent_dims = 4
    num_epochs = 2
    batch_size = 128
    capacity = 64
    learning_rate = 1e-3
    variational_beta = 1
    alpha = 1e-6
    use_gpu = True
    validate_every_n_iterations = 200

    device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")

    do_dequantize = True
    
    img_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_dataset = MNIST(root='./data/MNIST', download=True, train=True, transform=img_transform)

    p_validation = 0.1
    size_validation = round(p_validation * len(train_dataset))
    size_train = len(train_dataset) - size_validation

    train_subset, val_subset = torch.utils.data.random_split(train_dataset, [size_train, size_validation],
                                                             generator=torch_generator)
    train_dataloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    validation_dataloader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

    test_dataset = MNIST(root='./data/MNIST', download=True, train=False, transform=img_transform)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    core_flow = RealNVP(input_dim=4, num_flows=6, hidden_units=256)
    encoder = Encoder(64,4)
    decoder = Decoder(64, 4, [1,28,28])
    mask = torch.zeros(28,28)
    mask[13:15, 13:15] = 1
    mask = mask.to(device)
    preprocessing_layers = [nf.transforms.Logit(alpha), nf.flows.ActNorm([1,28,28])]
    nae = NormalizingAutoEncoder(core_flow, encoder, decoder, mask, preprocessing_layers)
    optimizer = torch.optim.Adam(params=nae.parameters(), lr=1e-3)#, weight_decay=1e-5)

    nae = nae.to(device)

    train_loss_avg = []

    print('Training ...')

    batch_bar = tqdm(train_dataloader, leave=False, desc='batch',
                     total=len(train_dataloader))
    for image_batch, _ in batch_bar:
        if do_dequantize:
            image_batch = dequantize(image_batch)
        image_batch = image_batch.to(device)

    print('Training ...')

    stop = False
    n_iterations_done = 0
    n_times_validated = 0
    iteration_losses = np.zeros((n_iterations, ))
    validation_losses = np.zeros(((n_iterations // validate_every_n_iterations) + 1, ))
    averaging_window_size = 10

    with tqdm(total=n_iterations, desc="iteration [loss: ...]") as iterations_bar:
        while not stop:
            for image_batch, _ in train_dataloader:
                if do_dequantize:
                    image_batch = dequantize(image_batch)
                image_batch = image_batch.to(device)

                loss = torch.mean(nae.neg_log_likelihood(image_batch))
                iteration_losses[n_iterations_done] = loss.item()

                # backpropagation
                optimizer.zero_grad()
                loss.backward()

                # one step of the optmizer
                optimizer.step()

                refresh_bar(iterations_bar,
                            f"iteration [loss: "
                            f"{get_avg_loss_over_iterations(iteration_losses, averaging_window_size, n_iterations_done):.3f}]")

                # We validate first epoch
                if (n_iterations_done % validate_every_n_iterations) == 0 or (n_iterations_done == n_iterations - 1):
                    val_batch_bar = tqdm(validation_dataloader, leave=False, desc='validation batch',
                                         total=len(validation_dataloader))
                    val_loss_averager = make_averager()
                    with torch.no_grad():
                        for validation_batch, _ in val_batch_bar:
                            if do_dequantize:
                                validation_batch = dequantize(validation_batch)
                            validation_batch = validation_batch.to(device)
                            loss = torch.mean(nae.neg_log_likelihood(validation_batch))

                            refresh_bar(val_batch_bar,
                                        f"validation batch [loss: {val_loss_averager(loss.item()):.3f}]")
                        validation_losses[n_times_validated] = val_loss_averager(None)
                        n_times_validated += 1

                n_iterations_done += 1
                iterations_bar.update(1)
                if n_iterations_done >= n_iterations:
                    stop = True
                    break


    plot_loss_over_iterations(iteration_losses, validation_losses, np.arange(n_times_validated) * validate_every_n_iterations)


    samples = nae.sample(16)
    samples = samples.cpu().detach().numpy()
    _, axs = plt.subplots(4, 4, )
    axs = axs.flatten()
    for img, ax in zip(samples, axs):
        ax.axis('off')
        ax.imshow(img.reshape(28, 28), cmap='gray')

    plt.show()

if __name__ == "__main__":
    main()
