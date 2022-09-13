import os
from typing import List

import torch.distributions
import torchvision

import numpy as np
from torch import Tensor
from torch.utils.data import DataLoader

from analysis import get_field_from_config
from datasets import get_test_dataloader
import util
import wandb
from models import model_database
from models.autoencoder_base import AutoEncoder, GaussianAutoEncoder
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.manifold import TSNE

from models.model_database import get_model
from util import load_best_model

from datetime import date

import traceback


def get_z_values(n_vals: int = 20, border: float = 0.10, latent_dims: int = 2):
    ''' Get z values needed to plot a grid of samples from the latent space. Grid over two dimensional z. '''
    lin_vals = torch.linspace(1-border,border, steps=n_vals)
    lin_vals_1 = torch.linspace(border,1-border, steps=n_vals)

    icdf_vals = torch.cartesian_prod(*([lin_vals, lin_vals_1]))
    distr = torch.distributions.normal.Normal(torch.zeros(latent_dims), torch.ones(latent_dims))
    z_vals = distr.icdf(icdf_vals)

    return torch.index_select(z_vals, 1, torch.tensor([1,0]))


def plot_latent_space_2d(model: AutoEncoder, test_loader, device, equal_axes=True, max_val=None, colorbar=True,
                         add_prior_flow=False):
    '''
    :param model:
    :param test_loader:
    :param device:
    :param max_val: max value from 0 in x and y direction: sometimes we see outliers
    :return:
    '''
    arr = np.zeros([len(test_loader.dataset), 3])
    n_added = 0
    for image_batch, image_labels in test_loader:
        image_batch = util.dequantize(image_batch)
        image_batch = image_batch.to(device)
        with torch.no_grad():
            output = model.encode(image_batch)
            if isinstance(output, tuple):
                mu, _ = model.encode(image_batch)
            else:
                mu = output
            if add_prior_flow:
                mu, _ = model.prior_bijector.inverse(mu)
        arr[n_added:n_added + len(image_batch), :2] = mu.cpu().detach().numpy()
        arr[n_added:n_added + len(image_batch), 2] = image_labels
        n_added += len(image_batch)
    plt.rcParams['axes.axisbelow'] = True
    fig = plt.figure(figsize=(6, 6), dpi=300)
    ax = fig.gca()

    # plt.style.use('seaborn')
    scat = plt.scatter(arr[:, 0], arr[:, 1], s=10, c=arr[:, 2], cmap=plt.get_cmap('tab10'), alpha=0.8, rasterized=True,
                       linewidths=0)
    # cb = plt.colorbar(scat, spacing='uniform', ticks=np.linspace(0, 9, 10))

    ax.set_facecolor('lavender')
    ax.grid(visible=True, which='major', axis='both', color='w', )

    # sns.set_theme()

    if equal_axes:
        plt.axis('equal')
    if max_val is not None:
        # cur_min_x, cur_max_x = np.min(arr[:, 0]), np.max(arr[:, 0])
        # cur_min_y, cur_max_y = np.min(arr[:, 1]), np.max(arr[:, 1])
        #
        # cur_min = min(cur_min_x, cur_min_y)
        # cur_max = max(cur_max_x, cur_max_y)
        #
        # if cur_min < -max_val or cur_max > max_val:
        #     lim = max_val
        # else:
        #     lim = max(-1 * cur_min, cur_max)
        lim = max_val
        # plt.ylim((max(cur_min_x, -max_val), min(cur_max_x, max_val)))  # Why are these reversed?
        # plt.xlim((max(cur_min_y, -max_val), min(cur_max_y, max_val)))

        plt.ylim((-lim, lim))  # Why are these reversed?
        plt.xlim((-lim, lim))
    else:
        left, right = plt.xlim()
        max_lr = max(-left, right)
        plt.xlim((-max_lr, max_lr))

    if colorbar:
        plt.clim(-0.5, 10 - 0.5)
        cb = plt.colorbar(scat, ticks=range(0, 10), spacing='uniform')
        cb.ax.tick_params(length=0)

    return fig


def plot_samples(model: AutoEncoder, img_shape: List = [1, 28, 28], n_rows: int = 10, n_cols: int = 10,
                 batch_size: int = 100, temperature: int = 1, padding: int = 1):
    '''
    Function to plot a grid of samples given a model.
    '''
    n_samples = n_rows * n_cols
    arr = torch.zeros((n_samples, *img_shape))
    n_filled = 0
    while n_filled < n_samples:
        n_to_sample = min(batch_size, n_samples - n_filled)
        with torch.no_grad():
            arr[n_filled:n_filled + n_to_sample] = model.sample(n_to_sample, temperature=temperature).cpu().detach()
        n_filled += n_to_sample

    arr = np.clip(arr, 0., 1.)
    grid = torchvision.utils.make_grid(arr, padding=padding, pad_value=0., nrow=n_cols, normalize=False)
    img = torchvision.transforms.ToPILImage()(grid)
    return img

def plot_reconstructions(model: GaussianAutoEncoder, test_loader: DataLoader, device: torch.device,
                         img_shape: List = [1, 28, 28], n_rows: int = 4, n_cols: int = 4, skip_batches=0, padding: int = 1):
    '''
    Function to plot a grid (size n_rows x n_rows) of reconstructions given a model. Each roww of original samples is
    followed by a row of reconstructions.
    '''

    n_images = n_rows * n_cols
    arr = torch.zeros((n_images, *img_shape))

    cur_row = 0
    iter_test_loader = iter(test_loader)

    n_images_filled = 0
    batches_skipped = 0
    while cur_row < n_rows:
        while batches_skipped <= skip_batches:
            image_batch, _ = next(iter_test_loader)
            batches_skipped += 1
        batch_idx = 0
        n_imgs_in_batch_left = image_batch.shape[0]
        while n_imgs_in_batch_left >= n_cols and cur_row < n_rows:
            n_imgs_in_batch_left -= n_cols  # We use the first n_cols images of the batch
            row_batch = image_batch[batch_idx:batch_idx + n_cols]
            arr[n_images_filled:n_images_filled + n_cols] = row_batch
            batch_idx += n_cols
            n_images_filled += n_cols
            row_batch = util.dequantize(row_batch)

            row_batch = row_batch.to(device)
            with torch.no_grad():

                z = model.encode(row_batch)
                if isinstance(z, tuple):
                    z = z[0]
                reconstruction = model.decode(z)
                # NAE returns a single value, VAEs will return mu and sigma
                if isinstance(reconstruction, tuple):
                    reconstruction = reconstruction[0]
                reconstruction = reconstruction.cpu().detach()
            arr[n_images_filled:n_images_filled + n_cols] = reconstruction
            n_images_filled += n_cols
            cur_row += 2  # We filled two rows

    arr = np.clip(arr, 0., 1.)
    grid = torchvision.utils.make_grid(arr, padding=padding, pad_value=0., nrow=n_cols, normalize=False)
    img = torchvision.transforms.ToPILImage()(grid)
    return img

def plot_noisy_reconstructions(model: GaussianAutoEncoder, image_batch: Tensor, device: torch.device,
                               noise_distribution: torch.distributions.Distribution,
                               img_shape: List = [1, 28, 28], n_rows: int = 6, n_cols: int = 6):
    '''
    Function to plot a grid (size n_rows x n_rows) of reconstructions given a model. Following row structure:
    1) image with noise 2) denoised image 3) original image
    '''
    n_images = n_rows * n_cols
    arr = torch.zeros((n_images, *img_shape))

    assert n_rows % 3 == 0
    assert n_images <= image_batch.shape[0]

    cur_row = 0
    n_images_filled = 0
    n_images_out_of_batch = 0

    while cur_row < n_rows:
        n_imgs_in_batch_left = image_batch.shape[0]
        while n_imgs_in_batch_left >= n_cols and cur_row < n_rows:
            n_imgs_in_batch_left -= n_cols  # We use the first n_cols images of the batch

            row_batch = image_batch[n_images_out_of_batch:n_images_out_of_batch + n_cols]
            n_images_out_of_batch += n_cols
            noisy_batch = torch.clone(row_batch).detach()
            noise = noise_distribution.sample()[:n_cols]  # What would be faster: this or reinitializing a
            # distribution of proper size each time?
            noisy_batch += noise
            noisy_batch = torch.clamp(noisy_batch, 0., 1.)

            # Fill noisy images
            arr[n_images_filled:n_images_filled + n_cols] = noisy_batch

            n_images_filled += n_cols

            noisy_batch = noisy_batch.to(device)
            with torch.no_grad():
                z = model.encode(noisy_batch)
                if isinstance(z, tuple):
                    z = z[0]
                reconstruction = model.decode(z)
                reconstruction = reconstruction.cpu().detach()
            # Fill reconstructions
            arr[n_images_filled:n_images_filled + n_cols] = reconstruction
            n_images_filled += n_cols

            # Fill originals
            arr[n_images_filled:n_images_filled + n_cols] = row_batch
            n_images_filled += n_cols

            cur_row += 3  # We filled three rows

    arr = np.clip(arr, 0., 1.)
    grid = torchvision.utils.make_grid(arr, padding=1, pad_value=0., nrow=n_rows)
    img = torchvision.transforms.ToPILImage()(grid)
    return img




