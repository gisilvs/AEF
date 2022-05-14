import os
from typing import List

import torch.distributions
import torchvision

import numpy as np
from torch.utils.data import DataLoader

from datasets import get_test_dataloader
import util
import wandb
from models.autoencoder_base import AutoEncoder, GaussianAutoEncoder
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from models.model_database import get_model
from util import load_best_model

from datetime import date

import traceback


def get_z_values(n_vals: int = 20, border: float = 0.15, latent_dims: int = 2):
    ''' Get z values needed to plot a grid of samples from the latent space. Grid over two dimensional z. '''
    lin_vals = torch.linspace(1-border,border, steps=n_vals)
    lin_vals_1 = torch.linspace(border,1-border, steps=n_vals)

    icdf_vals = torch.cartesian_prod(*([lin_vals, lin_vals_1]))
    distr = torch.distributions.normal.Normal(torch.zeros(latent_dims), torch.ones(latent_dims))
    z_vals = distr.icdf(icdf_vals)

    return torch.index_select(z_vals, 1, torch.tensor([1,0]))


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


def plot_latent_space_2d(model: AutoEncoder, test_loader, device, equal_axes=True, max_val=None):
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
        arr[n_added:n_added + len(image_batch), :2] = mu.cpu().detach().numpy()
        arr[n_added:n_added + len(image_batch), 2] = image_labels
        n_added += len(image_batch)
    plt.rcParams['axes.axisbelow'] = True
    fig = plt.figure(figsize=(6, 6))
    ax = fig.gca()

    #plt.style.use('seaborn')
    scat = plt.scatter(arr[:, 0], arr[:, 1], s=10, c=arr[:, 2], cmap=plt.get_cmap('tab10'), alpha=0.8)
    cb = plt.colorbar(scat, spacing='uniform')
    ax.set_facecolor('lavender')
    ax.grid(visible=True, which='major', axis='both', color='w', )

    # sns.set_theme()

    if equal_axes:
        plt.axis('equal')
    if max_val is not None:
        cur_min_x, cur_max_x = np.min(arr[:, 0]), np.max(arr[:, 0])
        cur_min_y, cur_max_y = np.min(arr[:, 1]), np.max(arr[:, 1])

        plt.ylim((max(cur_min_x, -max_val), min(cur_max_x, max_val)))  # Why are these reversed?
        plt.xlim((max(cur_min_y, -max_val), min(cur_max_y, max_val)))
    return fig


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
                               noise_distribution: torch.distributions.Distribution,
                               img_shape: List = [1, 28, 28], n_rows: int = 6, hires=False,
                               image_batch=None):
    '''
    Function to plot a grid (size n_rows x n_rows) of reconstructions given a model. Following row structure:
    1) image with noise 2) denoised image 3) original image
    '''
    n_cols = n_rows
    n_images = n_rows * n_cols
    arr = torch.zeros((n_images, *img_shape))

    cur_row = 0
    iter_test_loader = iter(test_loader)

    n_images_filled = 0

    noise_to_device = image_batch is not None

    while cur_row < n_rows:
        if image_batch is None:
            image_batch, _ = next(iter_test_loader)

        batch_idx = 0
        n_imgs_in_batch_left = image_batch.shape[0]
        while n_imgs_in_batch_left >= n_cols and cur_row < n_rows:
            n_imgs_in_batch_left -= n_cols  # We use the first n_cols images of the batch

            row_batch = image_batch[batch_idx:batch_idx + n_cols]
            batch_idx += n_cols

            noisy_batch = torch.clone(row_batch).detach()
            noise = noise_distribution.sample()[:n_rows] # What would be faster: this or reinitializing a
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
        fig = plt.figure(figsize=(10, 10))
    grid_img = torchvision.utils.make_grid(arr, padding=1, pad_value=0., nrow=n_rows)
    plt.imshow(grid_img.permute(1, 2, 0))
    plt.axis("off")
    return fig

def generate_2d_grids():
    datasets = ['mnist', 'fashionmnist', 'kmnist']
    model_names = ['vae', 'iwae', 'vae-iaf', 'nae-center', 'nae-corner', 'nae-external']
    #model_names = ['nae-center', 'nae-corner', 'nae-external']
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
                    ''' latent_x = np.linspace(-1.5, 1.5, 20)
                    latent_y = np.linspace(-1.5, 1.5, 20)
                    latents = torch.FloatTensor(len(latent_y), len(latent_x), 2)
                    for i, lx in enumerate(latent_x):
                        for j, ly in enumerate(latent_y):
                            latents[j, i, 0] = lx
                            latents[j, i, 1] = ly
                    latents = latents.view(-1, 2)  # flatten grid into a batch

                    z_vals = latents.to(device)'''
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
    datasets = ['mnist', 'fashionmnist', 'kmnist']
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

                fig = plot_latent_space_2d(model, test_loader, device, equal_axes=True, max_val=5)
                plt.savefig(f'plots/latent_{run_name}.pdf', bbox_inches='tight', pad_inches=0)
                z_vals = get_z_values(n_vals=latent_grid_size, latent_dims=2, border=0.06)
                z_vals = z_vals.to(device)

                with torch.no_grad():
                    z_vals, _ = model.prior_bijector.forward(z_vals)
                    output = model.decode(z_vals)
                    if isinstance(output, tuple):
                        output = output[0]
                    output = output.detach().cpu()

                fig = util.plot_image_grid(output, cols=latent_grid_size, padding=0, hires=True)
                plt.savefig(f'plots/grid_{run_name}.pdf', transparent='true', bbox_inches='tight', pad_inches=0)
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
                                                    #"config.posterior_flow": posterior_flow,
                                                    #"config.prior_flow": prior_flow
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

                fig = plot_latent_space_2d(model, test_loader, device, equal_axes=True, max_val=5)
                plt.savefig(f'plots/latent_{run_name}.pdf', bbox_inches='tight', pad_inches=0)
                z_vals = get_z_values(n_vals=latent_grid_size, latent_dims=2, border=0.06)
                z_vals = z_vals.to(device)

                with torch.no_grad():

                    output = model.decode(z_vals)
                    if isinstance(output, tuple):
                        output = output[0]
                    output = output.detach().cpu()

                fig = util.plot_image_grid(output, cols=latent_grid_size, padding=0, hires=True)
                plt.savefig(f'plots/grid_{run_name}.pdf', bbox_inches='tight', pad_inches=0)
            except Exception as E:
                print(E)
                print(f'Failed to plot latent space of {experiment_name}')
                traceback.print_exc()
                continue
        plt.close('all')
    # visualization_run.finish()

def generate_denoising_reconstructions():
    datasets = ['fashionmnist']
    model_name = 'nae-external'
    posterior_flow = 'none'
    prior_flow = 'none'

    #posterior_flow = 'maf'
    #prior_flow = 'maf'
    decoder = 'independent'
    # model_name = 'vae'
    # posterior_flow = 'iaf'
    # prior_flow = 'maf'
    # decoder = 'fixed'
    latent_dims = 32
    api = wandb.Api()
    architecture_size = 'small'
    img_dim = [1, 28, 28]
    alpha = 0.05
    project_name = 'denoising-experiments-1'
    noise_level = 0.75

    use_gpu = True
    device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")
    latent_grid_size = 20
    # visualization_run = wandb.init(project='visualizations', entity="nae", name=run_name)
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
                noise_distribution = torch.distributions.normal.Normal(torch.zeros([6, *img_dim]),
                                                                       noise_level * torch.ones([6, *img_dim]))
                fig = plot_noisy_reconstructions(model, test_loader, device, noise_distribution,
                                                 img_dim, n_rows=6)
                plt.savefig(f'plots/denoising_{run_name}.pdf', bbox_inches='tight', pad_inches=0)
            except Exception as E:
                print(E)
                print(f'Failed to plot latent space of {experiment_name}')
                traceback.print_exc()
                continue
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
                    fig=plot_reconstructions(model, test_loader, device, img_dim, n_rows=4, skip_batches=i)
                    plt.savefig(f'plots/cifar_{run_name}_{i}.pdf', bbox_inches='tight', pad_inches=0)
            except Exception as E:
                print(E)
                print(f'Failed to plot latent space of {experiment_name}')
                traceback.print_exc()
                continue
        plt.close('all')
    # visualization_run.finish()

def generate_samples():
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

    run_name = 'samples'

    visualization_run = wandb.init(project='visualizations', entity="nae", name=run_name)
    for latent in latent_dims:
        for dataset in datasets:
            for model_name in model_names:
                if 'nae' in model_name:
                    decoder = 'independent'
                else:
                    decoder = 'fixed'
                runs = api.runs(path="nae/phase1", filters={"config.dataset": dataset,
                                                            "config.latent_dims": latent,
                                                            "config.model": model_name,
                                                            })

                for run in runs:
                    run_id = run.id
                    experiment_name = run.name
                    try:

                        posterior_flow = 'none'
                        prior_flow = 'none'
                        model = get_model(model_name, architecture_size, decoder, latent, img_dim, alpha,
                                          posterior_flow,
                                          prior_flow)
                        run_name = run.name
                        artifact = api.artifact(
                            f'nae/{project_name}/{run_name}_best:latest')  # run.restore(f'{run_name}_best:latest', run_path=run.path, root='./artifacts')
                        artifact_dir = artifact.download()
                        artifact_dir = artifact_dir + '/' + os.listdir(artifact_dir)[0]
                        model.load_state_dict(torch.load(artifact_dir, map_location=device))
                        model = model.to(device)

                        for i in range(2):
                            fig = plot_samples(model, img_dim)
                            wandb.log(
                                {f"samples {dataset} {model_name} latent dim {latent} run {run_id} plot {i}": wandb.Image(fig)})
                        plt.close('all')
                    except Exception as E:
                        print(E)
                        print(f'Failed to plot samples of {experiment_name}')
                        traceback.print_exc()
                        continue
    visualization_run.finish()


def generate_visualizations_separately():
    generate_visualizations(False, True, False, False, 0, custom_name="2D latent space plots new")
    # generate_visualizations(False, False, True, False, 0, custom_name="Latent grid plots")
    # generate_visualizations(False, False, False, True, 5, custom_name="Samples")
    #generate_visualizations(False, False, False, False, 5, True, 1, custom_name="Reconstructions_new")

def generate_reconstructions(run_name = None):
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
    for latent in latent_dims:
        for dataset in datasets:
            for model_name in model_names:
                if 'nae' in model_name:
                    decoder = 'independent'
                else:
                    decoder = 'fixed'
                runs = api.runs(path="nae/phase1", filters={"config.dataset": dataset,
                                                            "config.latent_dims": latent,
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


                        test_loader = get_test_dataloader(dataset)
                        for i in range(3):
                            fig = plot_reconstructions(model, test_loader, device, img_dim, n_rows=10, skip_batches=i)
                            wandb.log(
                                {f"reconstructions {dataset} {model_name} latent {latent }run {run_id}": wandb.Image(fig)})
                        plt.close('all')
                    except Exception as E:
                        print(E)
                        print(f'Failed to plot reconstructions of {experiment_name}')
                        traceback.print_exc()
                        continue
    visualization_run.finish()

def generate_2d_latent_spaces(run_name = None):
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

def generate_visualizations(do_plot_latent_space_greater_than_2=False,
                            do_plot_latent_space_equal_to_2=True,
                            do_plot_samples_from_latent_space_grid=True,
                            do_plot_samples=True,
                            n_times_to_sample=5,
                            do_plot_reconstructions=True,
                            n_times_to_reconstruct=5,
                            custom_name=None):  # Number of plots to create for each run):
    '''5
    Function to create all plots for all runs of a certain phase. These will all be under a single wandb run.
    For latent grid search: latent grid {dataset} {model_name} run {run_nr}
    For latent space search: latent dims {latent_size} {dataset} {model_name} run {run_nr}
    For samples search: samples {latent_size} {dataset} {model_name} run {run_nr} plot {i}
    :return:
    '''
    today = date.today()
    project_name = 'phase1'

    latent_sizes = [2, 4, 8, 16]

    alpha = 1e-6
    use_gpu = True
    device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")

    latent_grid_size = 20
    datasets = ['mnist', 'fashionmnist']

    models = ['nae-center', 'nae-corner', 'nae-external', 'vae', 'vae-iaf', 'iwae']

    run_name = f"{today}_all" if custom_name is None else custom_name

    n_reconstruction_rows = 10

    architecture_size = 'small'
    prior_flow = 'none'
    posterior_flow = 'none'

    run = wandb.init(project='visualizations', entity="nae", name=run_name)
    for run_nr in range(5):
        for dataset in datasets:
            if 'mnist' in dataset:
                image_dim = [1, 28, 28]
            elif dataset == 'cifar' or dataset == 'cifar10':
                image_dim = [3, 32, 32]
            else:
                print("Unknown dataset. Quitting.")
                return
            for latent_size in latent_sizes:
                latent_dims = latent_size
                for model_name in models:
                    if model_name == 'nae-corner':
                        experiment_name = f'nae_{dataset}_run_{run_nr}_latent_size_{latent_size}_decoder_independent_corner'
                        decoder = 'independent'
                        model_name = 'nae-corner'
                    elif model_name == 'nae-center':
                        experiment_name = f'nae_{dataset}_run_{run_nr}_latent_size_{latent_size}_decoder_independent_center'
                        decoder = 'independent'
                        model_name = 'nae-center'
                    elif model_name == 'nae-external':
                        experiment_name = f'nae-external_{dataset}_run_{run_nr}_latent_size_{latent_size}_decoder_independent'
                        decoder = 'independent'
                        model_name = 'nae-external'
                    else:
                        experiment_name = f'{model_name}_{dataset}_run_{run_nr}_latent_size_{latent_size}_decoder_fixed'
                        decoder = 'fixed'

                    try:
                        model = load_best_model(run, project_name, model_name, experiment_name, device, latent_dims,
                                                image_dim,
                                                alpha, decoder, architecture_size, posterior_flow, prior_flow,
                                                version='latest')
                    except Exception as E:
                        print(E)
                        traceback.print_exc()
                        continue
                    model = model.to(device)

                    if latent_size == 2:
                        if do_plot_samples_from_latent_space_grid:
                            z_vals = get_z_values(n_vals=latent_grid_size, latent_dims=2)
                            z_vals = z_vals.to(device)

                            with torch.no_grad():
                                output = model.decode(z_vals)[0].detach().cpu()

                            util.plot_image_grid(output, cols=latent_grid_size, padding=0, hires=True)
                            image_dict = {f'latent grid {dataset} {model_name} run {run_nr}': plt}
                            wandb.log({**image_dict})
                        if do_plot_latent_space_equal_to_2:
                            test_loader = get_test_dataloader(dataset)
                            fig = plot_latent_space(model, test_loader, device)
                            wandb.log(
                                {f"latent dims {latent_size} {dataset} {model_name} run {run_nr}": wandb.Image(fig)})
                    else:
                        # Don't do these for >2d latent space, doesn't add that much since very dependent on TSNE.
                        if do_plot_latent_space_greater_than_2:
                            test_loader = get_test_dataloader(dataset)
                            fig = plot_latent_space(model, test_loader, device)
                            wandb.log(
                                {f"latent dims {latent_size} {dataset} {model_name} run {run_nr}": wandb.Image(fig)})
                    if do_plot_samples:
                        for i in range(n_times_to_sample):
                            fig = plot_samples(model, image_dim)
                            wandb.log({
                                          f"samples {latent_size} {dataset} {model_name} run {run_nr} plot {i}": wandb.Image(
                                              fig)})
                    if do_plot_reconstructions:
                        test_loader = get_test_dataloader(dataset, shuffle=True)
                        for i in range(n_times_to_reconstruct):
                            fig = plot_reconstructions(model, test_loader, device, image_dim, n_reconstruction_rows)
                            wandb.log({
                                          f"Reconstructions {latent_size} {dataset} {model_name} run {run_nr} plot {i}": wandb.Image(
                                              fig)})
                    plt.close('all')
    run.finish()



if __name__ == "__main__":
    #generate_pics_nae_external()
    generate_denoising_reconstructions()
    #generate_2d_latent_spaces('samples transparent')
    #generate_visualizations_separately()
    # generate_loss_over_latentdims()