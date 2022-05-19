import os
from typing import List

import torch.distributions
import torchvision

import numpy as np
from torch.utils.data import DataLoader

from analysis import get_field_from_config
from datasets import get_test_dataloader
import util
import wandb
from models import model_database
from models.autoencoder_base import AutoEncoder, GaussianAutoEncoder
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from models.model_database import get_model
from util import load_best_model

from datetime import date

import traceback


def get_z_values(n_vals: int = 20, border: float = 0.15, latent_dims: int = 2):
    ''' Get z values needed to plot a grid of samples from the latent space. Grid over two dimensional z. '''
    lin_vals = torch.linspace(1 - border, border, steps=n_vals)
    lin_vals_1 = torch.linspace(border, 1 - border, steps=n_vals)

    icdf_vals = torch.cartesian_prod(*([lin_vals, lin_vals_1]))
    distr = torch.distributions.normal.Normal(torch.zeros(latent_dims), torch.ones(latent_dims))
    z_vals = distr.icdf(icdf_vals)

    return torch.index_select(z_vals, 1, torch.tensor([1, 0]))


def plot_latent_space(model, test_loader, device):
    ''' Function to generate a plot of the latent space of a model. If there are more than 2 dimensions we use
    TSNE. '''
    n_latent_dims = model.encoder.latent_dim
    if n_latent_dims == 2:
        return plot_latent_space_2d(model, test_loader, device)

    arr = np.zeros((len(test_loader.dataset), n_latent_dims))
    labels = np.zeros((len(test_loader.dataset),))
    n_added = 0
    for image_batch, image_labels in test_loader:
        image_batch = util.dequantize(image_batch)
        image_batch = image_batch.to(device)
        mu, _ = model.encode(image_batch)
        arr[n_added:n_added + len(image_batch), :] = mu.cpu().detach().numpy()
        labels[n_added:n_added + len(image_batch)] = image_labels
        n_added += len(image_batch)

    embedded = TSNE(learning_rate='auto', init='pca', perplexity=50).fit_transform(
        arr)  # TSNE(learning_rate='auto', perplexity=50, n_iter=2000, init='pca').fit_transform(arr)

    fig = plt.figure(figsize=(10, 10))
    plt.style.use('seaborn')
    scat = plt.scatter(embedded[:, 0], embedded[:, 1], s=10, c=labels, cmap=plt.get_cmap('tab10'))
    cb = plt.colorbar(scat, spacing='proportional')
    plt.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=False)  # labels along the bottom edge are off
    plt.tick_params(
        axis='y',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        left=False,  # ticks along the bottom edge are off
        right=False,  # ticks along the top edge are off
        labelleft=False)  # labels along the bottom edge are off
    return fig


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


def save_colorbar(model, test_loader, device):
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
        arr[n_added:n_added + len(image_batch), :2] = mu.cpu().detach().numpy()
        arr[n_added:n_added + len(image_batch), 2] = image_labels
        n_added += len(image_batch)
    plt.rcParams['axes.axisbelow'] = True
    fig = plt.figure(figsize=(6, 6), dpi=300)
    ax = fig.gca()

    # plt.style.use('seaborn')
    fig, ax = plt.subplots(figsize=(6, 6), dpi=300)
    scat = plt.scatter(arr[:, 0], arr[:, 1], s=10, c=arr[:, 2], cmap=plt.get_cmap('tab10'), alpha=0.8, rasterized=True,
                       linewidths=0)

    plt.clim(-0.5, 10 - 0.5)
    cb = plt.colorbar(scat, ticks=range(0, 10), spacing='uniform', ax=ax)
    cb.ax.tick_params(length=0)
    ax.remove()
    # plt.savefig('plots/plot_onlycbar.png', dpi=400)

    # save the same figure with some approximate autocropping
    plt.savefig('plots/plot_onlycbar_tight.pdf', bbox_inches='tight', dpi=400)


def plot_samples(model: AutoEncoder, img_shape: List = [1, 28, 28], n_rows: int = 10, n_cols: int = 10,
                 batch_size: int = 100, temperature: int = 1):
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

    fig = plt.figure(figsize=(10, 10))
    grid_img = torchvision.utils.make_grid(arr, padding=1, pad_value=0., nrow=n_rows)
    plt.imshow(grid_img.permute(1, 2, 0))
    plt.axis("off")
    return fig


def plot_reconstructions(model: GaussianAutoEncoder, test_loader: DataLoader, device: torch.device,
                         img_shape: List = [1, 28, 28], n_rows: int = 4, skip_batches=0):
    '''
    Function to plot a grid (size n_rows x n_rows) of reconstructions given a model. Each roww of original samples is
    followed by a row of reconstructions.
    '''
    n_cols = n_rows
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

    fig = plt.figure(figsize=(10, 10))
    grid_img = torchvision.utils.make_grid(arr, padding=1, pad_value=0., nrow=n_rows)
    plt.imshow(grid_img.permute(1, 2, 0))
    plt.axis("off")
    return fig


def plot_noisy_reconstructions(model: GaussianAutoEncoder, test_loader: DataLoader, device: torch.device,
                               noise_distribution: torch.distributions.Distribution = None,
                               img_shape: List = [1, 28, 28], n_rows: int = 6, n_cols: int = 6, hires=False,
                               image_batch=None, skip_batches: int = 0):
    '''
    Function to plot a grid (size n_rows x n_rows) of reconstructions given a model. Following row structure:
    1) image with noise 2) denoised image 3) original image
    '''
    n_images = n_rows * n_cols
    arr = torch.zeros((n_images, *img_shape))

    cur_row = 0
    iter_test_loader = iter(test_loader)

    n_images_filled = 0

    noise_to_device = image_batch is not None

    while cur_row < n_rows:
        if image_batch is None:
            skipped = 0
            while skipped < skip_batches:
                image_batch, _ = next(iter_test_loader)
                skipped += 1

        batch_idx = 0
        n_imgs_in_batch_left = image_batch.shape[0]
        while n_imgs_in_batch_left >= n_cols and cur_row < n_rows:
            n_imgs_in_batch_left -= n_cols  # We use the first n_cols images of the batch

            row_batch = image_batch[batch_idx:batch_idx + n_cols]
            batch_idx += n_cols

            noisy_batch = torch.clone(row_batch).detach()
            noise = noise_distribution.sample()[:n_cols]  # What would be faster: this or reinitializing a
            # distribution of proper size each time?
            if noise_to_device:
                noise = noise.to(device)
            noisy_batch += noise
            noisy_batch = torch.clamp(noisy_batch, 0., 1.)
            # Fill noisy images
            arr[n_images_filled:n_images_filled + n_cols] = noisy_batch

            n_images_filled += n_cols
            # row_batch = util.dequantize(row_batch)

            noisy_batch = noisy_batch.to(device)
            with torch.no_grad():
                encode_output = model.encode(noisy_batch)
                if isinstance(encode_output, tuple):
                    z, _ = encode_output
                else:
                    z = encode_output
                reconstruction = model.decode(z)
                # NAE returns a single value, VAEs will return mu and sigma
                if isinstance(reconstruction, tuple):
                    reconstruction = reconstruction[0]
                reconstruction = reconstruction.cpu().detach()
            # Fill reconstructions
            arr[n_images_filled:n_images_filled + n_cols] = reconstruction
            n_images_filled += n_cols

            # Fill originals
            arr[n_images_filled:n_images_filled + n_cols] = row_batch
            n_images_filled += n_cols

            cur_row += 3  # We filled three rows
    if hires:
        fig = plt.figure(figsize=(10, 10))
    else:
        fig = plt.figure(figsize=(6, 3))
    grid_img = torchvision.utils.make_grid(arr, padding=1, pad_value=0., nrow=n_cols)
    plt.imshow(grid_img.permute(1, 2, 0))
    plt.axis("off")
    return fig


def generate_2d_grids():
    datasets = ['mnist', 'fashionmnist', 'kmnist']
    model_names = ['vae', 'iwae', 'vae-iaf', 'nae-center', 'nae-corner', 'nae-external']
    # model_names = ['nae-center', 'nae-corner', 'nae-external']
    latent_dims = 2
    api = wandb.Api()
    architecture_size = 'small'
    img_dim = [1, 28, 28]
    alpha = 1e-6
    project_name = 'phase1'

    use_gpu = True
    device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")

    run_name = 'latent grids 2 latent dims'

    latent_grid_size = 20

    visualization_run = wandb.init(project='visualizations', entity="nae", name=run_name)
    for dataset in datasets:
        for model_name in model_names:
            if 'nae' in model_name:
                decoder = 'independent'
            else:
                decoder = 'fixed'
            runs = api.runs(path="nae/phase1", filters={"config.dataset": dataset,
                                                        "config.latent_dims": latent_dims,
                                                        "config.model": model_name,
                                                        })

            for run in runs:
                run_id = run.id
                experiment_name = run.name
                try:

                    posterior_flow = 'none'
                    prior_flow = 'none'
                    model = get_model(model_name, architecture_size, decoder, latent_dims, img_dim, alpha,
                                      posterior_flow,
                                      prior_flow)
                    run_name = run.name
                    artifact = api.artifact(
                        f'nae/{project_name}/{run_name}_best:latest')  # run.restore(f'{run_name}_best:latest', run_path=run.path, root='./artifacts')
                    artifact_dir = artifact.download()
                    artifact_dir = artifact_dir + '/' + os.listdir(artifact_dir)[0]
                    model.load_state_dict(torch.load(artifact_dir, map_location=device))
                    model = model.to(device)

                    z_vals = get_z_values(n_vals=latent_grid_size, latent_dims=2, border=0.06)
                    z_vals = z_vals.to(device)

                    with torch.no_grad():
                        output = model.decode(z_vals)
                        if isinstance(output, tuple):
                            output = output[0]
                        output = output.detach().cpu()

                    fig = util.plot_image_grid(output, cols=latent_grid_size, padding=0, hires=True)
                    wandb.log(
                        {f"latent grid {dataset} {model_name} run {run_id}": wandb.Image(fig)})
                    plt.close('all')
                except Exception as E:
                    print(E)
                    print(f'Failed to plot latent space of {experiment_name}')
                    traceback.print_exc()
                    continue
    visualization_run.finish()


def generate_pics_vae_maf_iaf():
    datasets = ['mnist', 'fashionmnist']  # , 'kmnist']
    model_name = 'vae'
    posterior_flow = 'iaf'
    prior_flow = 'maf'
    decoder = 'fixed'
    latent_dims = 2
    api = wandb.Api()
    architecture_size = 'small'
    img_dim = [1, 28, 28]
    alpha = 1e-6
    project_name = 'phase1'

    use_gpu = True
    device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")

    run_name = 'latent grids vae iaf maf'

    latent_grid_size = 20

    # visualization_run = wandb.init(project='visualizations', entity="nae", name=run_name)
    for dataset in datasets:

        runs = api.runs(path="nae/phase1", filters={"config.dataset": dataset,
                                                    "config.latent_dims": latent_dims,
                                                    "config.model": model_name,
                                                    "config.posterior_flow": "iaf",
                                                    "config.prior_flow": "maf"
                                                    })

        for run in runs:
            run_id = run.id
            experiment_name = run.name
            try:

                model = get_model(model_name, architecture_size, decoder, latent_dims, img_dim, alpha,
                                  posterior_flow,
                                  prior_flow)
                run_name = run.name
                artifact = api.artifact(
                    f'nae/{project_name}/{run_name}_best:latest')  # run.restore(f'{run_name}_best:latest', run_path=run.path, root='./artifacts')
                artifact_dir = artifact.download()
                artifact_dir = artifact_dir + '/' + os.listdir(artifact_dir)[0]
                model.load_state_dict(torch.load(artifact_dir, map_location=device))
                model = model.to(device)

                test_loader = get_test_dataloader(dataset)

                # fig = plot_latent_space_2d(model, test_loader, device, equal_axes=True)
                # plt.savefig(f'plots/latent_{run_name}.pdf', dpi=400, bbox_inches='tight')

                fig = plot_latent_space_2d(model, test_loader, device, equal_axes=True, max_val=3.5, colorbar=False,
                                           add_prior_flow=True)
                plt.savefig(f'plots/prior_latent_{run_name}_no_colorbar.pdf', dpi=400, bbox_inches='tight')
                z_vals = get_z_values(n_vals=latent_grid_size, latent_dims=2, border=0.06)
                z_vals = z_vals.to(device)

                # with torch.no_grad():
                #     z_vals, _ = model.prior_bijector.forward(z_vals)
                #     output = model.decode(z_vals)
                #     if isinstance(output, tuple):
                #         output = output[0]
                #     output = output.detach().cpu()
                #
                # fig = util.plot_image_grid(output, cols=latent_grid_size, hires=True, padding=0)
                # plt.savefig(f'plots/grid_{run_name}.pdf', dpi=400, transparent='true', bbox_inches='tight', pad_inches=0)
            except Exception as E:
                print(E)
                print(f'Failed to plot latent space of {experiment_name}')
                traceback.print_exc()
                continue
        plt.close('all')
    # visualization_run.finish()


def generate_pics_nae_external():
    datasets = ['mnist', 'fashionmnist']
    model_name = 'nae-external'
    posterior_flow = 'none'
    prior_flow = 'none'
    decoder = 'independent'
    latent_dims = 2
    api = wandb.Api()
    architecture_size = 'small'
    img_dim = [1, 28, 28]
    alpha = 1e-6
    project_name = 'phase1'

    use_gpu = True
    device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")
    latent_grid_size = 20
    # visualization_run = wandb.init(project='visualizations', entity="nae", name=run_name)
    for dataset in datasets:

        runs = api.runs(path="nae/phase1", filters={"config.dataset": dataset,
                                                    "config.latent_dims": latent_dims,
                                                    "config.model": model_name,
                                                    # "config.posterior_flow": posterior_flow,
                                                    # "config.prior_flow": prior_flow
                                                    })

        for run in runs:
            run_id = run.id
            experiment_name = run.name
            try:

                model = get_model(model_name, architecture_size, decoder, latent_dims, img_dim, alpha,
                                  posterior_flow,
                                  prior_flow)
                run_name = run.name
                artifact = api.artifact(
                    f'nae/{project_name}/{run_name}_best:latest')  # run.restore(f'{run_name}_best:latest', run_path=run.path, root='./artifacts')
                artifact_dir = artifact.download()
                artifact_dir = artifact_dir + '/' + os.listdir(artifact_dir)[0]
                model.load_state_dict(torch.load(artifact_dir, map_location=device))
                model = model.to(device)

                test_loader = get_test_dataloader(dataset)

                # fig = plot_latent_space_2d(model, test_loader, device, equal_axes=True, max_val=3.5)
                # plt.savefig(f'plots/latent_{run_name}.png', dpi=400, bbox_inches='tight')

                fig = plot_latent_space_2d(model, test_loader, device, equal_axes=True, max_val=3.5, colorbar=False)
                plt.savefig(f'plots/latent_{run_name}_no_colorbar.pdf', dpi=400, bbox_inches='tight')
                z_vals = get_z_values(n_vals=latent_grid_size, latent_dims=2, border=0.06)
                z_vals = z_vals.to(device)

                with torch.no_grad():

                    output = model.decode(z_vals)
                    if isinstance(output, tuple):
                        output = output[0]
                    output = output.detach().cpu()

                fig = util.plot_image_grid(output, cols=latent_grid_size, padding=0, hires=True)
                plt.savefig(f'plots/grid_{run_name}.pdf', bbox_inches='tight', dpi=400, pad_inches=0)
            except Exception as E:
                print(E)
                print(f'Failed to plot latent space of {experiment_name}')
                traceback.print_exc()
                continue
        plt.close('all')
    save_colorbar(model, test_loader, device)
    # visualization_run.finish()

def generate_latent_spaces():
    datasets = ['mnist']
    model_names = ['vae-iaf-maf'] #'nae-external', 'vae', 'vae-iaf',
    latent_dims = 2
    api = wandb.Api()
    architecture_size = 'small'
    img_dim = [1, 28, 28]
    alpha = 1e-6
    project_name = 'phase1'

    use_gpu = True
    device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")

    add_prior_flow = False
    for dataset in datasets:
        for model_name in model_names:
            if model_name == 'vae-iaf-maf':
                model_name = 'vae'
                runs = api.runs(path=f"nae/{project_name}",
                                filters={"config.dataset": dataset,
                                         "config.latent_dims": latent_dims,
                                         "config.model": 'vae',
                                         "config.posterior_flow": 'iaf',
                                         "config.prior_flow": 'maf'
                                         })
                add_prior_flow = True
            else:

                runs = api.runs(path=f"nae/{project_name}",
                                filters={"config.dataset": dataset,
                                         "config.latent_dims": latent_dims,
                                         "config.model": model_name,
                                         })
                add_prior_flow = False
            for run in runs:
                run_id = run.id
                experiment_name = run.name
                try:
                    posterior_flow = get_field_from_config(run, 'posterior_flow')
                    if posterior_flow is None:
                        posterior_flow = 'none'
                    prior_flow = get_field_from_config(run, 'prior_flow')
                    if prior_flow is None:
                        prior_flow = 'none'

                    decoder = get_field_from_config(run, 'decoder')
                    model = get_model(model_name, architecture_size, decoder, latent_dims, img_dim, alpha,
                                      posterior_flow,
                                      prior_flow)
                    run_name = run.name
                    artifact = api.artifact(
                        f'nae/{project_name}/{run_name}_best:latest')  # run.restore(f'{run_name}_best:latest', run_path=run.path, root='./artifacts')
                    artifact_dir = artifact.download()
                    artifact_dir = artifact_dir + '/' + os.listdir(artifact_dir)[0]
                    model.load_state_dict(torch.load(artifact_dir, map_location=device))
                    model = model.to(device)

                    test_loader = get_test_dataloader(dataset)

                    fig = plot_latent_space_2d(model, test_loader, device, equal_axes=True, max_val=3.5, add_prior_flow=add_prior_flow)
                    plt.savefig(f'plots/latent_spaces/{run_name}.png', dpi=400, bbox_inches='tight')

                    fig = plot_latent_space_2d(model, test_loader, device, equal_axes=True, max_val=3.5, colorbar=False, add_prior_flow=add_prior_flow)
                    plt.savefig(f'plots/latent_spaces/{run_name}_no_colorbar.pdf', dpi=400, bbox_inches='tight')

                except Exception as E:
                    print(E)
                    print(f'Failed to plot latent space of {experiment_name}')
                    traceback.print_exc()
                    continue
            plt.close('all')
    save_colorbar(model, test_loader, device)

def generate_denoising_reconstructions_main():
    datasets = ['fashionmnist']
    models = ['ae', 'vae-iaf-maf', 'nae-external']
    model_name = 'ae'
    posterior_flow = 'none'
    prior_flow = 'none'

    # posterior_flow = 'maf'
    # prior_flow = 'maf'
    decoder = 'fixed'
    # model_name = 'vae'
    # posterior_flow = 'iaf'
    # prior_flow = 'maf'
    # decoder = 'fixed'
    latent_dims = 32
    api = wandb.Api()
    architecture_size = 'small'
    img_dim = [1, 28, 28]
    alpha = 1e-6
    project_name = 'denoising-experiments-1'
    noise_level = 0.75

    use_gpu = True
    device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")
    latent_grid_size = 20
    # visualization_run = wandb.init(project='visualizations', entity="nae", name=run_name)
    for model_name in models:
        for dataset in datasets:
            runs = api.runs(path=f"nae/{project_name}", filters={"config.dataset": dataset,
                                                                 "config.latent_dims": latent_dims,
                                                                 "config.model": model_name,
                                                                 # "config.posterior_flow": posterior_flow,
                                                                 # "config.prior_flow": prior_flow,
                                                                 "config.noise_level": noise_level
                                                                 })

            for run in runs:
                run_id = run.id
                experiment_name = run.name
                try:
                    decoder = get_field_from_config(run, 'decoder')

                    model = model_database.get_model_denoising(model_name, decoder, latent_dims, img_dim, alpha)
                    run_name = run.name
                    artifact = api.artifact(
                        f'nae/{project_name}/{run_name}_best:latest')  # run.restore(f'{run_name}_best:latest', run_path=run.path, root='./artifacts')
                    artifact_dir = artifact.download()
                    artifact_dir = artifact_dir + '/' + os.listdir(artifact_dir)[0]
                    model.load_state_dict(torch.load(artifact_dir, map_location=device))
                    model = model.to(device)

                    test_loader = get_test_dataloader(dataset)
                    torch.manual_seed(3)
                    noise_distribution = torch.distributions.normal.Normal(torch.zeros([60, *img_dim]),
                                                                           noise_level * torch.ones([60, *img_dim]))
                    fig = plot_noisy_reconstructions(model, test_loader, device, noise_distribution,
                                                     img_dim, n_rows=3, n_cols=6)
                    plt.savefig(f'plots/denoising_{run_name}.pdf', bbox_inches='tight', pad_inches=0)
                except Exception as E:
                    print(E)
                    print(f'Failed to plot latent space of {experiment_name}')
                    traceback.print_exc()
                    continue
            plt.close('all')

def generate_denoising_reconstructions_supp():
    datasets = ['fashionmnist']
    models = ['ae', 'vae-iaf-maf', 'nae-external']
    model_name = 'ae'
    posterior_flow = 'none'
    prior_flow = 'none'

    # posterior_flow = 'maf'
    # prior_flow = 'maf'
    decoder = 'fixed'
    # model_name = 'vae'
    # posterior_flow = 'iaf'
    # prior_flow = 'maf'
    # decoder = 'fixed'
    latent_dims = 32
    api = wandb.Api()
    architecture_size = 'small'
    img_dim = [1, 28, 28]
    alpha = 1e-6
    project_name = 'denoising-experiments-1'
    noise_levels = [0.25, 0.5, 0.75]

    use_gpu = True
    device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")
    latent_grid_size = 20
    # visualization_run = wandb.init(project='visualizations', entity="nae", name=run_name)
    for model_name in models:
        for dataset in datasets:
            for noise_level in noise_levels:
                runs = api.runs(path=f"nae/{project_name}", filters={"config.dataset": dataset,
                                                                     "config.latent_dims": latent_dims,
                                                                     "config.model": model_name,
                                                                     # "config.posterior_flow": posterior_flow,
                                                                     # "config.prior_flow": prior_flow,
                                                                     "config.noise_level": noise_level
                                                                     })

                for run in runs:
                    run_id = run.id
                    experiment_name = run.name
                    try:
                        decoder = get_field_from_config(run, 'decoder')

                        model = model_database.get_model_denoising(model_name, decoder, latent_dims, img_dim, alpha)
                        run_name = run.name
                        artifact = api.artifact(
                            f'nae/{project_name}/{run_name}_best:latest')  # run.restore(f'{run_name}_best:latest', run_path=run.path, root='./artifacts')
                        artifact_dir = artifact.download()
                        artifact_dir = artifact_dir + '/' + os.listdir(artifact_dir)[0]
                        model.load_state_dict(torch.load(artifact_dir, map_location=device))
                        model = model.to(device)

                        test_loader = get_test_dataloader(dataset)

                        torch.manual_seed(3)
                        noise_distribution = torch.distributions.normal.Normal(torch.zeros([60, *img_dim]),
                                                                               noise_level * torch.ones([60, *img_dim]))
                        for i in range(5, 8):
                            fig = plot_noisy_reconstructions(model, test_loader, device, noise_distribution,
                                                             img_dim, n_rows=6, n_cols=6, skip_batches=i)
                        plt.savefig(f'plots/denoising/{run_name}_{i}.pdf', bbox_inches='tight', pad_inches=0)
                    except Exception as E:
                        print(E)
                        print(f'Failed to plot latent space of {experiment_name}')
                        traceback.print_exc()
                        continue
                    break
                plt.close('all')


def generate_cifar_reconstructions():
    datasets = ['cifar']
    model_name = 'nae-external'
    posterior_flow = 'maf'
    prior_flow = 'maf'
    decoder = 'independent'
    # model_name = 'vae'
    # posterior_flow = 'iaf'
    # prior_flow = 'maf'
    # decoder = 'fixed'
    latent_dims = 64
    api = wandb.Api()
    architecture_size = 'small'
    img_dim = [3, 32, 32]
    alpha = 0.05
    project_name = 'cifar'

    use_gpu = True
    device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")
    latent_grid_size = 20
    # visualization_run = wandb.init(project='visualizations', entity="nae", name=run_name)
    for dataset in datasets:

        runs = api.runs(path=f"nae/{project_name}", filters={"config.dataset": dataset,
                                                             "config.latent_dims": latent_dims,
                                                             "config.model": model_name,
                                                             "config.posterior_flow": posterior_flow,
                                                             "config.prior_flow": prior_flow
                                                             })

        for run in runs:
            run_id = run.id
            experiment_name = run.name
            try:

                model = get_model(model_name, architecture_size, decoder, latent_dims, img_dim, alpha,
                                  posterior_flow,
                                  prior_flow)
                run_name = run.name
                artifact = api.artifact(
                    f'nae/{project_name}/{run_name}_best:latest')  # run.restore(f'{run_name}_best:latest', run_path=run.path, root='./artifacts')
                artifact_dir = artifact.download()
                artifact_dir = artifact_dir + '/' + os.listdir(artifact_dir)[0]
                model.load_state_dict(torch.load(artifact_dir, map_location=device))
                model = model.to(device)

                test_loader = get_test_dataloader(dataset)
                for i in range(20, 23):
                    fig = plot_reconstructions(model, test_loader, device, img_dim, n_rows=4, skip_batches=i)
                    plt.savefig(f'plots/cifar_{run_name}_{i}.pdf', bbox_inches='tight', pad_inches=0)
            except Exception as E:
                print(E)
                print(f'Failed to plot latent space of {experiment_name}')
                traceback.print_exc()
                continue
        plt.close('all')
    # visualization_run.finish()




def generate_reconstructions(run_name=None):
    datasets = ['mnist', 'fashionmnist', 'kmnist']
    model_names = ['vae', 'iwae', 'vae-iaf', 'nae-center', 'nae-corner', 'nae-external']
    latent_dims = [2, 4, 8, 16, 32]
    api = wandb.Api()
    architecture_size = 'small'
    img_dim = [1, 28, 28]
    alpha = 1e-6
    project_name = 'phase1'

    use_gpu = True
    device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")

    run_name = 'reconstructions'

    visualization_run = wandb.init(project='visualizations', entity="nae", name=run_name)
    for latent_dim in latent_dims:
        for dataset in datasets:
            for model_name in model_names:
                if 'nae' in model_name:
                    decoder = 'independent'
                else:
                    decoder = 'fixed'
                runs = api.runs(path="nae/phase1", filters={"config.dataset": dataset,
                                                            "config.latent_dims": latent_dim,
                                                            "config.model": model_name,
                                                            })

                for run in runs:
                    run_id = run.id
                    experiment_name = run.name
                    try:

                        posterior_flow = 'none'
                        prior_flow = 'none'
                        model = get_model(model_name, architecture_size, decoder, latent_dim, img_dim, alpha,
                                          posterior_flow,
                                          prior_flow)
                        run_name = run.name
                        artifact = api.artifact(
                            f'nae/{project_name}/{run_name}_best:latest')  # run.restore(f'{run_name}_best:latest', run_path=run.path, root='./artifacts')
                        artifact_dir = artifact.download()
                        artifact_dir = artifact_dir + '/' + os.listdir(artifact_dir)[0]
                        model.load_state_dict(torch.load(artifact_dir, map_location=device))
                        model = model.to(device)

                        test_loader = get_test_dataloader(dataset)
                        for i in range(3):
                            fig = plot_reconstructions(model, test_loader, device, img_dim, n_rows=10, skip_batches=i)
                            wandb.log(
                                {f"reconstructions {dataset} {model_name} latent {latent_dim}run {run_id}": wandb.Image(
                                    fig)})
                        plt.close('all')
                    except Exception as E:
                        print(E)
                        print(f'Failed to plot reconstructions of {experiment_name}')
                        traceback.print_exc()
                        continue
    visualization_run.finish()


def generate_2d_latent_spaces(run_name=None):
    datasets = ['mnist', 'fashionmnist', 'kmnist']
    model_names = ['vae', 'iwae', 'vae-iaf', 'nae-center', 'nae-corner', 'nae-external']
    latent_dims = 2
    api = wandb.Api()
    architecture_size = 'small'
    img_dim = [1, 28, 28]
    alpha = 1e-6
    project_name = 'phase1'

    use_gpu = True
    device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")
    if run_name is None:
        run_name = 'latent spaces 2 latent dims'

    visualization_run = wandb.init(project='visualizations', entity="nae", name=run_name)
    for dataset in datasets:
        for model_name in model_names:
            if 'nae' in model_name:
                decoder = 'independent'
            else:
                decoder = 'fixed'
            runs = api.runs(path="nae/phase1", filters={"config.dataset": dataset,
                                                        "config.latent_dims": latent_dims,
                                                        "config.model": model_name,
                                                        })

            for run in runs:
                if run.state != 'finished':
                    continue
                run_id = run.id
                experiment_name = run.name
                try:

                    posterior_flow = 'none'
                    prior_flow = 'none'
                    model = get_model(model_name, architecture_size, decoder, latent_dims, img_dim, alpha,
                                      posterior_flow,
                                      prior_flow)
                    run_name = run.name
                    artifact = api.artifact(
                        f'nae/{project_name}/{run_name}_best:latest')  # run.restore(f'{run_name}_best:latest', run_path=run.path, root='./artifacts')
                    artifact_dir = artifact.download()
                    artifact_dir = artifact_dir + '/' + os.listdir(artifact_dir)[0]
                    model.load_state_dict(torch.load(artifact_dir, map_location=device))
                    model = model.to(device)

                    test_loader = get_test_dataloader(dataset)
                    fig = plot_latent_space(model, test_loader, device)
                    wandb.log(
                        {f"latent space {dataset} {model_name} run {run_id}": wandb.Image(fig)})
                    plt.close('all')
                except Exception as E:
                    print(E)
                    print(f'Failed to plot latent space of {experiment_name}')
                    traceback.print_exc()
                    continue
    visualization_run.finish()


def generate_phase1_samples():
    projects = ['phase1', 'cifar']
    datasets_projects = {'phase1': ['mnist', 'fashionmnist', 'kmnist'],
                'cifar': ['cifar']}
    model_names = ['nae-center', 'nae-corner', 'vae-iaf-maf']
    latent_sizes = {'phase1': [32], 'cifar': [64]}
    api = wandb.Api()
    architecture_size = 'small'
    img_dims_projects = {'phase1': [1, 28, 28], 'cifar': [3, 32, 32]}
    alpha_projects = {'phase1': 1e-6, 'cifar': 0.05}

    use_gpu = True
    device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")

    for project_name in projects:
        for dataset in datasets_projects[project_name]:
            for latent_dims in latent_sizes[project_name]:
                for model_name in model_names:
                    if model_name == 'vae-iaf-maf':
                        model_name = 'vae'
                        runs = api.runs(path=f"nae/{project_name}",
                                        filters={"config.dataset": dataset,
                                                 "config.latent_dims": latent_dims,
                                                 "config.model": 'vae',
                                                 "config.posterior_flow": 'iaf',
                                                 "config.prior_flow": 'maf'
                                                 })
                    else:

                        runs = api.runs(path=f"nae/{project_name}",
                                        filters={"config.dataset": dataset,
                                                 "config.latent_dims": latent_dims,
                                                 "config.model": model_name,
                                                 })
                    img_dims = img_dims_projects[project_name]
                    alpha = alpha_projects[project_name]

                    for run in runs:
                        run_id = run.id
                        experiment_name = run.name

                        try:

                            posterior_flow = get_field_from_config(run, 'posterior_flow')
                            if posterior_flow is None:
                                posterior_flow = 'none'
                            prior_flow = get_field_from_config(run, 'prior_flow')
                            if prior_flow is None:
                                prior_flow = 'none'

                            decoder = get_field_from_config(run, 'decoder')
                            model = get_model(model_name, architecture_size, decoder, latent_dims, img_dims, alpha,
                                              posterior_flow,
                                              prior_flow)
                            run_name = run.name
                            artifact = api.artifact(
                                f'nae/{project_name}/{run_name}_best:latest')  # run.restore(f'{run_name}_best:latest', run_path=run.path, root='./artifacts')
                            artifact_dir = artifact.download()
                            artifact_dir = artifact_dir + '/' + os.listdir(artifact_dir)[0]
                            model.load_state_dict(torch.load(artifact_dir, map_location=device))
                            model = model.to(device)

                            fig = plot_samples(model, img_dims, n_rows=8, n_cols=8, temperature=1)
                            plt.savefig(f'plots/samples/{run_name}.pdf', pad_inches=0, bbox_inches='tight')
                            plt.close('all')
                        except Exception as E:
                            print(E)
                            print(f'Failed to plot samples of {experiment_name}')
                            traceback.print_exc()
                            continue
                        break

def generate_phase1_reconstructions():
    projects = ['phase1', 'cifar']
    datasets_projects = {'phase1': ['mnist', 'fashionmnist', 'kmnist'],
                'cifar': ['cifar']}
    model_names = ['nae-center', 'nae-corner', 'vae-iaf-maf']
    latent_sizes = {'phase1': [32], 'cifar': [64]}
    api = wandb.Api()
    architecture_size = 'small'
    img_dims_projects = {'phase1': [1, 28, 28], 'cifar': [3, 32, 32]}
    alpha_projects = {'phase1': 1e-6, 'cifar': 0.05}

    use_gpu = True
    device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")

    for project_name in projects:
        for dataset in datasets_projects[project_name]:
            for latent_dims in latent_sizes[project_name]:
                for model_name in model_names:
                    if model_name == 'vae-iaf-maf':
                        model_name = 'vae'
                        runs = api.runs(path=f"nae/{project_name}",
                                        filters={"config.dataset": dataset,
                                                 "config.latent_dims": latent_dims,
                                                 "config.model": 'vae',
                                                 "config.posterior_flow": 'iaf',
                                                 "config.prior_flow": 'maf'
                                                 })
                    else:
                        runs = api.runs(path=f"nae/{project_name}",
                                    filters={"config.dataset": dataset,
                                             "config.latent_dims": latent_dims,
                                             "config.model": model_name,
                                             })
                    img_dims = img_dims_projects[project_name]
                    alpha = alpha_projects[project_name]

                    for run in runs:
                        run_id = run.id
                        experiment_name = run.name

                        try:

                            # posterior_flow = 'none'
                            # prior_flow = 'none'
                            posterior_flow = get_field_from_config(run, 'posterior_flow')
                            if posterior_flow is None:
                                posterior_flow = 'none'
                            prior_flow = get_field_from_config(run, 'prior_flow')
                            if prior_flow is None:
                                prior_flow = 'none'

                            decoder = get_field_from_config(run, 'decoder')
                            model = get_model(model_name, architecture_size, decoder, latent_dims, img_dims, alpha,
                                              posterior_flow,
                                              prior_flow)
                            run_name = run.name
                            artifact = api.artifact(
                                f'nae/{project_name}/{run_name}_best:latest')  # run.restore(f'{run_name}_best:latest', run_path=run.path, root='./artifacts')
                            artifact_dir = artifact.download()
                            artifact_dir = artifact_dir + '/' + os.listdir(artifact_dir)[0]
                            model.load_state_dict(torch.load(artifact_dir, map_location=device))
                            model = model.to(device)

                            test_loader = get_test_dataloader(dataset)
                            for i in range(2):

                                fig = plot_reconstructions(model, test_loader, device, img_dims, n_rows=8,
                                                               skip_batches=i)

                                plt.savefig(f'plots/reconstructions/{run_name}_{i}.pdf', pad_inches=0, bbox_inches='tight')
                            plt.close('all')
                        except Exception as E:
                            print(E)
                            print(f'Failed to plot samples of {experiment_name}')
                            traceback.print_exc()
                            continue
                        break

def generate_celeba_samples_main():
    fig, axs = plt.subplots(2, 3, figsize=(6.4, 4.8), dpi=300)

    model_names = ['nae-external', 'vae']
    latent_sizes = [64, 128, 256]

    project_name = 'phase2'

    architecture_size = 'big'
    img_dim = [3, 32, 32]
    alpha = 0.05

    api = wandb.Api()
    use_gpu = True
    device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")

    params = {
        'axes.titlesize': 'xx-large',
    }
    plt.rcParams.update(params)

    for model_idx, model_name in enumerate(model_names):
        for latent_idx, latent_dims in enumerate(latent_sizes):
            runs = api.runs(path="nae/phase2", filters={
                "config.latent_dims": latent_dims,
                "config.model": model_name,
            })
            for run in runs:
                run_id = run.id
                experiment_name = run.name

                model_name = get_field_from_config(run, "model")

                dataset = get_field_from_config(run, "dataset")

                decoder = get_field_from_config(run, "decoder")
                latent_dims = get_field_from_config(run, "latent_dims", type="int")

                posterior_flow = get_field_from_config(run, 'posterior_flow')
                prior_flow = get_field_from_config(run, 'prior_flow')

                model = get_model(model_name, architecture_size, decoder, latent_dims, img_dim, alpha,
                                  posterior_flow,
                                  prior_flow)
                run_name = run.name
                artifact = api.artifact(
                    f'nae/{project_name}/{run_name}_best:latest')  # run.restore(f'{run_name}_best:latest', run_path=run.path, root='./artifacts')
                artifact_dir = artifact.download()
                artifact_dir = artifact_dir + '/' + os.listdir(artifact_dir)[0]
                model.load_state_dict(torch.load(artifact_dir, map_location=device))
                model = model.to(device)

                samples = model.sample(4).detach().cpu()
                grid_img = torchvision.utils.make_grid(samples, padding=0, pad_value=0., nrow=2)
                axs[model_idx, latent_idx].imshow(grid_img.permute(1, 2, 0))
                axs[model_idx, latent_idx].axis("off")
                if model_idx == 0:
                    axs[model_idx, latent_idx].set_title(f'{latent_dims}')

                if latent_idx == 0:
                    lbl = 'AEF' if model_idx == 0 else 'VAE'
                    axs[model_idx, latent_idx].set_xlabel(lbl)
    fig.tight_layout()
    plt.savefig('plots/celeba_samples.pdf', dpi=300, bbox_inches='tight')
    plt.show()

def generate_celeba_samples_supp():
    model_names = ['nae-external', 'vae']
    latent_sizes = [64, 128, 256]

    project_name = 'phase2'

    architecture_size = 'big'
    img_dim = [3, 32, 32]
    alpha = 0.05

    api = wandb.Api()
    use_gpu = True
    device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")

    for model_idx, model_name in enumerate(model_names):
        for latent_idx, latent_dims in enumerate(latent_sizes):
            runs = api.runs(path="nae/phase2", filters={
                "config.latent_dims": latent_dims,
                "config.model": model_name,
            })
            for run_idx, run in enumerate(runs):
                run_id = run.id
                experiment_name = run.name

                model_name = get_field_from_config(run, "model")

                dataset = get_field_from_config(run, "dataset")

                decoder = get_field_from_config(run, "decoder")
                latent_dims = get_field_from_config(run, "latent_dims", type="int")

                posterior_flow = get_field_from_config(run, 'posterior_flow')
                prior_flow = get_field_from_config(run, 'prior_flow')

                model = get_model(model_name, architecture_size, decoder, latent_dims, img_dim, alpha,
                                  posterior_flow,
                                  prior_flow)
                run_name = run.name
                artifact = api.artifact(
                    f'nae/{project_name}/{run_name}_best:latest')  # run.restore(f'{run_name}_best:latest', run_path=run.path, root='./artifacts')
                artifact_dir = artifact.download()
                artifact_dir = artifact_dir + '/' + os.listdir(artifact_dir)[0]
                model.load_state_dict(torch.load(artifact_dir, map_location=device))
                model = model.to(device)
                for temp in [0.4]: #[0.4, 0.6, 0.8, 1.0]:
                    for i in range(3):
                        plot_samples(model, img_dim, n_rows=8, n_cols=8, batch_size=64, temperature=temp)
                        plt.savefig(f'plots/celeba_samples_{temp}_run_{run_idx}_{i}.pdf', dpi=300, bbox_inches='tight', pad_inches=0)

                plt.close("all")

def generate_celeba_reconstructions_supp():
    model_names = ['nae-external', 'vae']
    latent_sizes = [64, 128, 256]

    project_name = 'phase2'

    architecture_size = 'big'
    img_dim = [3, 32, 32]
    alpha = 0.05

    api = wandb.Api()
    use_gpu = True
    device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")
    k = 0
    for model_idx, model_name in enumerate(model_names):
        for latent_idx, latent_dims in enumerate(latent_sizes):
            runs = api.runs(path="nae/phase2", filters={
                "config.latent_dims": latent_dims,
                "config.model": model_name,
            })
            for run_idx, run in enumerate(runs):
                run_id = run.id
                experiment_name = run.name

                model_name = get_field_from_config(run, "model")

                dataset = get_field_from_config(run, "dataset")

                decoder = get_field_from_config(run, "decoder")
                latent_dims = get_field_from_config(run, "latent_dims", type="int")

                posterior_flow = get_field_from_config(run, 'posterior_flow')
                prior_flow = get_field_from_config(run, 'prior_flow')

                model = get_model(model_name, architecture_size, decoder, latent_dims, img_dim, alpha,
                                  posterior_flow,
                                  prior_flow)
                run_name = run.name
                artifact = api.artifact(
                    f'nae/{project_name}/{run_name}_best:latest')  # run.restore(f'{run_name}_best:latest', run_path=run.path, root='./artifacts')
                artifact_dir = artifact.download()
                artifact_dir = artifact_dir + '/' + os.listdir(artifact_dir)[0]
                model.load_state_dict(torch.load(artifact_dir, map_location=device))
                model = model.to(device)
                test_loader = get_test_dataloader(dataset, data_dir="data/celebahq")
                for i in range(2):
                    fig = plot_reconstructions(model, test_loader, device, img_dim, n_rows=8,
                                               skip_batches=k+i)

                    plt.savefig(f'plots/reconstructions/{run_name}_{i}.pdf', pad_inches=0, bbox_inches='tight')
                plt.close('all')


if __name__ == "__main__":
    # rc = {
    #     "text.usetex": True,
    #     "font.family": "Times New Roman",
    # }
    # plt.rcParams.update(rc)
    #generate_denoising_reconstructions()
    # generate_pics_nae_external()
    #generate_celeba_samples_supp()
    # generate_denoising_reconstructions()
    # generate_pics_vae_maf_iaf()
    # generate_phase1_samples()
    # generate_phase1_reconstructions()
    #generate_denoising_reconstructions_supp()
    # generate_celeba_reconstructions_supp()
    # generate_celeba_samples_supp()
    generate_latent_spaces()
    exit()
    # generate_denoising_reconstructions()
