import torch.distributions
import torchvision

import numpy as np

from datasets import get_test_dataloader
import util
import wandb
from models.autoencoder_base import AutoEncoder
from models.models import get_model
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from util import load_best_model


# def get_model_by_run_id(project: str, run_id: str, run_name:str, which_file: str = 'best', wandb_api: wandb.apis.public.Api = None):
#     if wandb_api is None:
#         api = wandb.Api()
#
#     run = api.run(f"nae/{project}/{run_id}")
#     config = run.config
#     model_name = config["model"]
#     latent_dims = config["latent_dims"]
#     dataset = config["dataset"]
#     image_dim = datasets.get_img_dims(dataset)
#     alpha = datasets.get_alpha(dataset)
#     use_center_pixels = True  # TODO: get from config or filename
#     device = 'cpu'
#
#     model = get_model(model_name, latent_dims, image_dim, alpha, use_center_pixels)
#     model.sample(10)  # needed as some components such as actnorm need to be initialized
#     artifact = api.artifact()
#     model_path = run.(run, project_name, experiment_name, version)
#     model.load_state_dict(torch.load(model_path, map_location=device))
#
#
# project = 'phase1'
# run_id = '3hho6lhe'
# get


def get_z_values(n_vals: int = 20, border: float = 0.15, latent_dims: int = 2):
    lin_vals = torch.linspace(border, 1 - border, steps=n_vals)
    icdf_vals = torch.cartesian_prod(*([lin_vals] * latent_dims))
    distr = torch.distributions.normal.Normal(torch.zeros(latent_dims), torch.ones(latent_dims))
    z_vals = distr.icdf(icdf_vals)

    return z_vals


def plot_latent_space(model, test_loader, device):
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
        arr[n_added:n_added + len(image_batch), :] = mu.detach().numpy()
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


def plot_latent_space_2d(model: AutoEncoder, test_loader, device):
    arr = np.zeros([len(test_loader.dataset), 3])
    n_added = 0
    for image_batch, image_labels in test_loader:
        image_batch = util.dequantize(image_batch)
        image_batch = image_batch.to(device)
        mu, _ = model.encode(image_batch)
        arr[n_added:n_added + len(image_batch), :2] = mu.detach().numpy()
        arr[n_added:n_added + len(image_batch), 2] = image_labels
        n_added += len(image_batch)

    fig = plt.figure(figsize=(10, 10), dpi=150)
    plt.style.use('seaborn')
    scat = plt.scatter(arr[:, 0], arr[:, 1], s=10, c=arr[:, 2], cmap=plt.get_cmap('tab10'))
    cb = plt.colorbar(scat, spacing='uniform')
    cur_min_x, cur_max_x = np.min(arr[:, 0]), np.max(arr[:, 0])
    cur_min_y, cur_max_y = np.min(arr[:, 1]), np.max(arr[:, 1])
    plt.ylim((max(cur_min_x, -5), min(cur_max_x, 5)))  # Why are these reversed?
    plt.xlim((max(cur_min_y, -5), min(cur_max_y, 5)))
    return fig

def generate_visualizations():

    run = wandb.init(project='visualizations', entity="nae")

    project_name = 'phase1'
    image_dim = [1, 28, 28]

    latent_sizes = [2, 4, 8, 16]

    alpha = 1e-6
    use_center_pixels = False
    use_gpu = False
    device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")

    latent_grid_size = 20
    datasets = ['mnist', 'fashionmnist']
    models = ['nae-center', 'nae-corner', 'vae', 'vae-iaf', 'iwae']  # ['vae', 'vae-iaf', 'iwae', 'nae']
    for dataset in datasets:
        for latent_size in latent_sizes:
            latent_dims = latent_size
            for model_name in models:
                if model_name == 'nae-corner':
                    experiment_name = f'nae_{dataset}_run_2_latent_size_{latent_size}_decoder_independent_corner'
                    decoder = 'independent'
                    model_name = 'nae'
                    model_name_addition = '-corner'
                    use_center_pixels = False
                elif model_name == 'nae-center':
                    experiment_name = f'nae_{dataset}_run_2_latent_size_{latent_size}_decoder_independent_center'
                    decoder = 'independent'
                    model_name = 'nae'
                    model_name_addition = '-center'
                    use_center_pixels = True
                else:
                    experiment_name = f'{model_name}_{dataset}_run_2_latent_size_{latent_size}_decoder_fixed'
                    decoder = 'fixed'
                    model_name_addition = ''  # This could be nicer

                try:
                    model = load_best_model(run, project_name, model_name, experiment_name, device, latent_dims, image_dim,
                                            alpha, decoder, use_center_pixels, version='best:latest')
                except Exception as E:
                    print(E)
                    continue
                model = model.to(device)

                if latent_size == 2:
                    z_vals = get_z_values(n_vals=latent_grid_size, latent_dims=2)
                    z_vals = z_vals.to(device)

                    output = model.sample(z=z_vals).detach().cpu()

                    util.plot_image_grid(output, cols=latent_grid_size, padding=0, hires=True)
                    image_dict = {f'latent grid {dataset} {model_name}{model_name_addition} ': plt}
                    wandb.log({**image_dict})

                test_loader = get_test_dataloader(dataset)
                fig = plot_latent_space(model, test_loader, device)
                wandb.log({f"latent dims {latent_size} {dataset} {model_name}{model_name_addition}": wandb.Image(fig)})
    # test_loader = datasets.get_test_dataloader(dataset)
    # plot_latent_space(model, test_loader)

        #
        # plt.show()
    run.finish()

def generate_loss_over_latentdims():
    api = wandb.Api(timeout=19)

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
    best_scores = -1 * np.nanmin(scores, axis=0)
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
    generate_loss_over_latentdims()