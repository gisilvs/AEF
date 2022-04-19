from typing import Mapping, Union, Optional
from pathlib import Path

import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.distributions import MultivariateNormal
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import torchvision
from torchvision import datasets, models, transforms
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from util import get_avg_loss_over_iterations

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

import os
import pickle
from tqdm import tqdm
# @title training utilities

import pandas as pd
import numpy as np

from typing import Callable, Optional

import util


def make_averager() -> Callable[[Optional[float]], float]:
    """ Returns a function that maintains a running average
    :returns: running average function
    """
    count = 0
    total = 0

    def averager(new_value: Optional[float]) -> float:
        """ Running averager
        :param new_value: number to add to the running average,
                          if None returns the current average
        :returns: the current average
        """
        nonlocal count, total
        if new_value is None:
            return total / count if count else float("nan")
        count += 1
        total += new_value
        return total / count

    return averager


def save_in_dataframe(df_log, labels, mus, stddevs, epoch):
    df = pd.DataFrame()

    df['index'] = np.arange(len(mus[:, 0])) * epoch
    df['image_ind'] = np.arange(len(mus[:, 0]))
    df['class'] = labels.data.numpy().astype(str)
    df['mu_x'] = mus[:, 0]
    df['mu_y'] = mus[:, 1]
    df['std_x'] = stddevs[:, 0]
    df['std_y'] = stddevs[:, 1]
    df['epoch'] = np.ones(len(mus[:, 0])) * epoch

    df_log = pd.concat([df_log, df])

    return df_log


def run_on_testbatch(df_log, vae, epoch, x, y):
    with torch.no_grad():
        x = x.to(device)
        x, mus, stddevs = vae(x)
        x = x.to('cpu')
        mus = mus.to('cpu').data.numpy()
        stddevs = stddevs.to('cpu').mul(0.5).exp_().data.numpy()

    return save_in_dataframe(df_log, y, mus, stddevs, epoch)


def plot_loss(train_loss, val_loss=None):
    plt.figure()
    plt.plot(np.arange(len(train_loss)), train_loss, color='tab:blue')
    if val_loss is not None:
        plt.plot(np.arange(len(val_loss)), val_loss, color='tab:orange')
    plt.legend(["Training loss"] if val_loss is None else ["Training loss", "Validation loss"])
    plt.xlabel("Epoch")
    plt.ylabel("Negative log likelihood")
    plt.show()


def refresh_bar(bar, desc):
    bar.set_description(desc)
    bar.refresh()


def dequantize(batch):
    noise = torch.rand(*batch.shape)
    batch = (batch * 255. + noise) / 256.
    return batch


def to_logit(batch, alpha):
    logit_batch = alpha + (1 - 2 * alpha) * batch
    return torch.log(logit_batch) - torch.log(1 - logit_batch)


def from_logit(logit_batch, alpha):
    logit_batch = torch.sigmoid(logit_batch)
    return (logit_batch - alpha) / (1 - 2 * alpha)


class NormalizingAutoEncoder(nn.Module):
    def __init__(self, core_flow, encoder, decoder, mask):
        super().__init__()

        self.core_flow = core_flow
        self.encoder = encoder
        self.decoder = decoder
        self.eps = 1e-5
        self.core_size = int(torch.sum(mask))
        self.mask = mask

    def partition(self, x):
        core = x[:, :, self.mask == 1].view(x.shape[0], -1)
        shell = x * (1 - self.mask)
        return core, shell

    def inverse_partition(self, core, shell):
        shell[:, :, self.mask == 1] = core.reshape(shell.shape[0], -1,
                                                   self.core_size)
        return shell

    def embedding(self, core, shell):
        core, log_j_core = self.core_flow.inverse(core)
        mu_z, sigma_z = self.encoder(shell)
        z = (core - mu_z) / (sigma_z + self.eps)
        log_j_z = torch.sum(-torch.log(sigma_z + self.eps), dim=[1])
        mu_d, sigma_d = self.decoder(z)
        deviations = (shell - mu_d) / (sigma_d + self.eps)
        log_j_d = torch.sum(-torch.log(sigma_d[:, :, self.mask == 0] + self.eps),
                            dim=[1, 2])
        return z, deviations, log_j_core + log_j_z + log_j_d

    def neg_log_likelihood(self, x):
        core, shell = self.partition(x)
        z, deviations, log_j = self.embedding(core, shell)
        loss_z = torch.sum(
            Normal(loc=torch.zeros_like(z), scale=torch.ones_like(z)).log_prob(
                z), dim=1)
        loss_d = torch.sum(Normal(loc=torch.zeros_like(deviations[:, :, self.mask == 0]),
                                  scale=torch.ones_like(deviations[:, :, self.mask == 0])).log_prob(
            deviations[:, :, self.mask == 0]), dim=[1, 2])
        return -(loss_z + loss_d + log_j)

    def sample(self, num_samples=1, sample_deviations=False):
        device = next(self.parameters()).device
        z = torch.normal(torch.zeros(num_samples, self.core_size),
                         torch.ones(num_samples, self.core_size)).to(device)
        if sample_deviations:
            deviations = torch.normal(torch.zeros_like(self.mask),
                                      torch.ones_like(self.mask)).to(device)  # TODO: change when refactoring
        else:
            deviations = None
        core, shell = self.forward(z, deviations)
        y = self.inverse_partition(core, shell)
        return y

    def forward(self, z, deviations=None):
        mu_d, sigma_d = self.decoder(z)
        if deviations is None:
            shell = mu_d
        else:
            shell = deviations * (sigma_d + self.eps) + mu_d
        shell = shell * (1 - self.mask)
        mu_z, sigma_z = self.encoder(shell)
        core = z * (sigma_z + self.eps) + mu_z
        core = self.core_flow.forward(core)
        return core, shell


class Encoder(nn.Module):
    def __init__(self, hidden_channels: int, latent_dim: int, input_channels: int = 1):
        """
        Simple encoder module
        It predicts the `mean` and `log(variance)` parameters.
        The choice to use the `log(variance)` is for stability reasons:
        https://stats.stackexchange.com/a/353222/284141
        """
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=input_channels,
                               out_channels=hidden_channels,
                               kernel_size=3,
                               stride=2,
                               padding=1)  # out: hidden_channels x 14 x 14

        self.conv2 = nn.Conv2d(in_channels=hidden_channels,
                               out_channels=hidden_channels * 2,
                               kernel_size=3,
                               stride=2,
                               padding=1)  # out: (hidden_channels x 2) x 7 x 7

        self.fc_mu = nn.Linear(in_features=hidden_channels * 2 * 7 * 7,
                               out_features=latent_dim)
        self.fc_logvar = nn.Linear(in_features=hidden_channels * 2 * 7 * 7,
                                   out_features=latent_dim)

        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor):
        """
        :param x: batch of images with shape [batch, channels, w, h]
        :returns: the predicted mean and log(variance)
        """
        batch_size = x.shape[0]
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))

        x = x.view(x.shape[0], -1)

        x_mu = self.fc_mu(x)
        x_mu = x_mu
        x_logvar = self.fc_logvar(x)
        x_logvar = x_logvar

        return x_mu.view(batch_size, -1), F.softplus(x_logvar).view(batch_size,
                                                                    -1)


class Decoder(nn.Module):
    def __init__(self, hidden_channels, latent_dim, output_shape):
        """
        Simple decoder module
        """
        super().__init__()

        self.output_shape = output_shape
        self.hidden_channels = hidden_channels

        self.fc = nn.Linear(in_features=latent_dim,
                            out_features=hidden_channels * 2 * 7 * 7)

        self.conv2 = nn.ConvTranspose2d(in_channels=hidden_channels * 2,
                                        out_channels=hidden_channels,
                                        kernel_size=4,
                                        stride=2,
                                        padding=1)
        self.conv1 = nn.ConvTranspose2d(in_channels=hidden_channels,
                                        out_channels=output_shape[0],
                                        kernel_size=4,
                                        stride=2,
                                        padding=1)

        self.activation = nn.ReLU()
        self.pre_sigma = nn.Parameter(torch.zeros(output_shape))

    def forward(self, x: torch.Tensor):
        """
        :param x: a sample from the distribution governed by the mean and log(var)
        :returns: a reconstructed image with size [batch, 1, w, h]
        """
        x = self.fc(x)
        x = x.view(x.size(0), self.hidden_channels * 2, 7, 7)
        x = self.activation(self.conv2(x))
        x = torch.sigmoid(self.conv1(
            x))  # last layer before output is sigmoid, since we are using BCE as reconstruction loss
        return x, F.softplus(self.pre_sigma).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)


class RealNVP(nn.Module):
    def __init__(self, input_dim, num_flows, hidden_units):
        super(RealNVP, self).__init__()

        nets = lambda: nn.Sequential(nn.Linear(input_dim, hidden_units),
                                     nn.LeakyReLU(),
                                     nn.Linear(hidden_units, hidden_units),
                                     nn.LeakyReLU(),
                                     nn.Linear(hidden_units, input_dim),
                                     nn.Tanh())
        nett = lambda: nn.Sequential(nn.Linear(input_dim, hidden_units),
                                     nn.LeakyReLU(),
                                     nn.Linear(hidden_units, hidden_units),
                                     nn.LeakyReLU(),
                                     nn.Linear(hidden_units, input_dim))

        masks = torch.from_numpy(np.asarray(
            [np.random.choice([0, 1], size=(input_dim,)) for _ in
             range(num_flows)]).astype(np.float32))
        prior = MultivariateNormal(torch.zeros(input_dim), torch.eye(input_dim))
        self.input_dim = input_dim
        self.prior = prior
        self.mask = nn.Parameter(masks, requires_grad=False)
        self.t = torch.nn.ModuleList([nett() for _ in range(num_flows)])
        self.s = torch.nn.ModuleList([nets() for _ in range(num_flows)])

    def g(self, z):
        x = z
        for i in range(len(self.t)):
            x_ = x * self.mask[i]
            s = self.s[i](x_) * (1 - self.mask[i])
            t = self.t[i](x_) * (1 - self.mask[i])
            x = x_ + (1 - self.mask[i]) * (x * torch.exp(s) + t)
        return x

    def f(self, x):
        log_det_J, z = x.new_zeros(x.shape[0]), x
        # print(log_det_J.shape)
        for i in reversed(range(len(self.t))):
            z_ = self.mask[i] * z
            s = self.s[i](z_) * (1 - self.mask[i])
            t = self.t[i](z_) * (1 - self.mask[i])
            z = (1 - self.mask[i]) * (z - t) * torch.exp(-s) + z_
            # print(s.shape)
            log_det_J -= s.sum(dim=[1])
        return z, log_det_J

    def inverse(self, x):
        return self.f(x)

    def forward(self, x):
        return self.g(x)

    def log_prob(self, x):
        z, logp = self.f(x)
        return self.prior.log_prob(z) + logp

    def sample(self, batchSize):
        z = self.prior.sample((batchSize, 1))
        logp = self.prior.log_prob(z)
        x = self.g(z)
        return x


def main():

    torch_generator = torch.Generator().manual_seed(3)

    do_dequantize = True
    do_logit_transform = False

    # 2-d latent space, parameter count in same order of magnitude
    # as in the original VAE paper (VAE paper has about 3x as many)
    latent_dims = 4
    n_iterations = 1000
    batch_size = 128
    capacity = 64
    learning_rate = 1e-3
    variational_beta = 1
    alpha = 1e-6
    use_gpu = True
    validate_every_n_iterations = 200

    device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")

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
    encoder = Encoder(64, 4)
    decoder = Decoder(64, 4, [1, 28, 28])
    mask = torch.zeros(28, 28)
    mask[13:15, 13:15] = 1
    mask = mask.to(device)
    nae = NormalizingAutoEncoder(core_flow, encoder, decoder, mask)
    optimizer = torch.optim.Adam(params=nae.parameters(), lr=1e-3)  # , weight_decay=1e-5)
    nae = nae.to(device)




    print('Training ...')

    stop = False
    n_iterations_done = 0
    iteration_losses = np.zeros((n_iterations, ))
    val_loss_avg = []
    averaging_window_size = 100

    with tqdm(total=n_iterations, desc="iteration [loss: ...]") as iterations_bar:
        while not stop:
            for image_batch, _ in train_dataloader:
                if do_dequantize:
                    image_batch = dequantize(image_batch)
                if do_logit_transform:
                    image_batch = to_logit(image_batch, alpha)
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

                if n_iterations_done % validate_every_n_iterations == 0:
                    val_batch_bar = tqdm(validation_dataloader, leave=False, desc='validation batch',
                                         total=len(validation_dataloader))
                    val_loss_averager = make_averager()
                    with torch.no_grad():
                        for validation_batch, _ in val_batch_bar:
                            if do_dequantize:
                                validation_batch = dequantize(validation_batch)
                            if do_logit_transform:
                                validation_batch = to_logit(validation_batch, alpha)
                            validation_batch = validation_batch.to(device)
                            loss = torch.mean(nae.neg_log_likelihood(validation_batch))

                            refresh_bar(val_batch_bar,
                                        f"validation batch [loss: {val_loss_averager(loss.item()):.3f}]")
                        val_loss_avg.append(val_loss_averager(None))

                n_iterations_done += 1
                iterations_bar.update(1)
                if n_iterations_done >= n_iterations:
                    stop = True
                    break

    samples = nae.sample(16)
    if do_logit_transform:
        samples = from_logit(samples, alpha)
    samples = samples.cpu().detach().numpy()
    _, axs = plt.subplots(4, 4, )
    axs = axs.flatten()
    for img, ax in zip(samples, axs):
        ax.axis('off')
        ax.imshow(img.reshape(28, 28), cmap='gray')

    plt.show()


if __name__ == "__main__":
    main()
