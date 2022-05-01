from typing import List

import torch.distributions
import torchvision

import numpy as np

from datasets import get_test_dataloader
import util
import wandb
from models.autoencoder_base import AutoEncoder
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from util import load_best_model

from datetime import date


def get_z_values(n_vals: int = 20, border: float = 0.15, latent_dims: int = 2):
    ''' Get z values needed to plot a grid of samples from the latent space. Grid over two dimensional z. '''
    lin_vals = torch.linspace(border, 1 - border, steps=n_vals)
    icdf_vals = torch.cartesian_prod(*([lin_vals] * latent_dims))
    distr = torch.distributions.normal.Normal(torch.zeros(latent_dims), torch.ones(latent_dims))
    z_vals = distr.icdf(icdf_vals)

    return z_vals


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

    fig = plt.figure(figsize=(10, 10), dpi=150)
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


def plot_latent_space_2d(model: AutoEncoder, test_loader, device, max_val=5):
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
            mu, _ = model.encode(image_batch)
        arr[n_added:n_added + len(image_batch), :2] = mu.cpu().detach().numpy()
        arr[n_added:n_added + len(image_batch), 2] = image_labels
        n_added += len(image_batch)

    fig = plt.figure(figsize=(10, 10), dpi=150)
    plt.style.use('seaborn')
    scat = plt.scatter(arr[:, 0], arr[:, 1], s=10, c=arr[:, 2], cmap=plt.get_cmap('tab10'))
    cb = plt.colorbar(scat, spacing='uniform')
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
            arr[n_filled:n_filled+n_to_sample] = model.sample(n_to_sample, temperature=temperature).cpu().detach()
        n_filled += n_to_sample

    fig = plt.figure(figsize=(10, 10), dpi=300)
    grid_img = torchvision.utils.make_grid(arr, padding=1, pad_value=0., nrow=n_rows)
    plt.imshow(grid_img.permute(1, 2, 0))
    plt.axis("off")
    return fig

def generate_visualizations_single_run():
    run_name = 'visualizations_XXX'
    project_name = 'phase1'
    model_name = 'nae-corner'
    experiment_name = f'nae_mnist_run_3_latent_size_4_decoder_independent_corner'
    decoder = 'independent'
    dataset = 'mnist'
    use_gpu = False
    device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")
    alpha = 1e-6
    latent_dims = 4
    image_dim = [1, 28, 28]

    run = wandb.init(project='visualizations', entity="nae", name=run_name)
    model = load_best_model(run, project_name, model_name, experiment_name, device, latent_dims, image_dim,
                            alpha, decoder, version='latest')

    do_plot_samples_from_latent_space_grid = False
    do_plot_latent_space_equal_to_2 = False
    latent_grid_size = 20
    do_plot_latent_space_greater_than_2 = False
    do_plot_samples = False
    n_times_to_sample = 5
    if do_plot_samples_from_latent_space_grid:
        z_vals = get_z_values(n_vals=latent_grid_size, latent_dims=2)
        z_vals = z_vals.to(device)

        with torch.no_grad():
            output = model.sample(z=z_vals).detach().cpu()

        util.plot_image_grid(output, cols=latent_grid_size, padding=0, hires=True)
        image_dict = {f'latent grid {dataset} {model_name}': plt}
        wandb.log({**image_dict})
    if do_plot_latent_space_equal_to_2:
        test_loader = get_test_dataloader(dataset)
        fig = plot_latent_space(model, test_loader, device)
        wandb.log({f"latent dims {latent_dims} {dataset} {model_name}": wandb.Image(fig)})


    # Don't do these for >2d latent space, doesn't add that much since very dependent on TSNE.
    if do_plot_latent_space_greater_than_2:
        test_loader = get_test_dataloader(dataset)
        fig = plot_latent_space(model, test_loader, device)
        wandb.log({f"latent dims {latent_dims} {dataset} {model_name}": wandb.Image(fig)})
    if do_plot_samples:
        for i in range(n_times_to_sample):
            fig = plot_samples(model, image_dim)
            wandb.log({f"samples {latent_dims} {dataset} {model_name} plot {i}": wandb.Image(fig)})



def generate_visualizations_separately():
    generate_visualizations(False, True, False, False, 0, custom_name="2D latent space plots")
    generate_visualizations(False, False, True, False, 0, custom_name="Latent grid plots")
    generate_visualizations(False, False, False, True, 5, custom_name="Samples")


def generate_visualizations(do_plot_latent_space_greater_than_2 = False,
                            do_plot_latent_space_equal_to_2 = True,
                            do_plot_samples_from_latent_space_grid = True,
                            do_plot_samples = True,
                            n_times_to_sample = 5,
                            custom_name = None): # Number of plots to create for each run):
    '''
    Function to create all plots for all runs of a certain phase. These will all be under a single wandb run.
    For latent grid search: latent grid {dataset} {model_name} run {run_nr}
    For latent space search: latent dims {latent_size} {dataset} {model_name} run {run_nr}
    For samples search: samples {latent_size} {dataset} {model_name} run {run_nr} plot {i}
    :return:
    '''
    today = date.today()
    project_name = 'phase1'
    image_dim = [1, 28, 28]
    latent_sizes = [2] #[2, 4, 8, 16]

    alpha = 1e-6
    use_gpu = True
    device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")

    latent_grid_size = 20
    datasets = ['mnist', 'fashionmnist']

    models = ['nae-center', 'nae-corner', 'nae-external', 'vae', 'vae-iaf', 'iwae']

    run_name = f"{today}_all" if custom_name is None else custom_name

    run = wandb.init(project='visualizations', entity="nae", name=run_name)
    for run_nr in range(5):
        for dataset in datasets:
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
                        model = load_best_model(run, project_name, model_name, experiment_name, device, latent_dims, image_dim,
                                                alpha, decoder, version='latest')
                    except Exception as E:
                        print(E)
                        continue
                    model = model.to(device)

                    if latent_size == 2:
                        if do_plot_samples_from_latent_space_grid:
                            z_vals = get_z_values(n_vals=latent_grid_size, latent_dims=2)
                            z_vals = z_vals.to(device)

                            with torch.no_grad():
                                output = model.sample(z=z_vals).detach().cpu()

                            util.plot_image_grid(output, cols=latent_grid_size, padding=0, hires=True)
                            image_dict = {f'latent grid {dataset} {model_name} run {run_nr}': plt}
                            wandb.log({**image_dict})
                        if do_plot_latent_space_equal_to_2:
                            test_loader = get_test_dataloader(dataset)
                            fig = plot_latent_space(model, test_loader, device)
                            wandb.log({f"latent dims {latent_size} {dataset} {model_name} run {run_nr}": wandb.Image(fig)})
                    else:
                        # Don't do these for >2d latent space, doesn't add that much since very dependent on TSNE.
                        if do_plot_latent_space_greater_than_2:
                            test_loader = get_test_dataloader(dataset)
                            fig = plot_latent_space(model, test_loader, device)
                            wandb.log({f"latent dims {latent_size} {dataset} {model_name} run {run_nr}": wandb.Image(fig)})
                    if do_plot_samples:
                        for i in range(n_times_to_sample):
                            fig = plot_samples(model, image_dim)
                            wandb.log({f"samples {latent_size} {dataset} {model_name} run {run_nr} plot {i}": wandb.Image(fig)})
    run.finish()


def generate_loss_over_latentdims():
    api = wandb.Api()

    project_name = 'phase1'
    image_dim = [1, 28, 28]

    alpha = 1e-6
    use_center_pixels = False
    use_gpu = False
    device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")

    dataset = 'mnist'
    model_name = 'nae'
    latent_sizes = [2, 4, 8, 16, 32]
    n_runs = 5
    decoder = 'independent'
    nae_type = 'corner'

    scores = np.zeros((n_runs, len(latent_sizes)))
    #n_recorded = np.zeros((len(latent_sizes)))
    for idx, latent_size in enumerate(latent_sizes):
        runs = api.runs(path="nae/phase1", filters={"config.dataset": dataset,
                                                    "config.latent_dims": latent_size,
                                                    "config.model": model_name,
                                                    "config.decoder": decoder,
                                                    })
        n_recorded = 0
        for run in runs:
            if nae_type not in run.name:
                continue
            if n_recorded >= n_runs:
                print(f'Found more than {n_runs} with latent size {latent_size}')
                break
            scores[n_recorded, idx] = run.summary['test_loss']
            n_recorded += 1
    scores[scores == np.inf] = np.nan
    best_scores = -1 * np.nanmin(scores, axis=0) # TODO: change to mean once inf/positive runs are gone
    fig = plt.figure(dpi=300)
    plt.scatter(np.arange(len(latent_sizes)), best_scores)
    plt.ylabel('Log-likelihood')
    plt.xlabel('Dimensionality of latent space')
    plt.xticks(np.arange(len(latent_sizes)), labels=[f'$2^{i+1}$' for i in range(len(latent_sizes))])

    # GET MAF likelihood
    scores_maf = np.zeros((n_runs,))
    runs = api.runs(path="nae/phase1", filters={"config.dataset": dataset,
                                                "config.model": 'maf',
                                                })
    n_recorded = 0
    for run in runs:
        if n_recorded >= n_runs:
            print(f'Found more than {n_runs} for maf')
            break
        scores_maf[n_recorded] = run.summary['test_loss']
        n_recorded += 1
    plt.axhline(y=-1 * np.min(scores_maf), color='k', linestyle='--')

    plt.savefig('./plots/likelihood_for_latents.png', dpi='figure')


if __name__ == "__main__":
    generate_visualizations_separately()
    #generate_loss_over_latentdims()