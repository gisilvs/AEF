import torch.distributions
import torchvision

import numpy as np

import datasets
import util
import wandb
from models.autoencoder_base import AutoEncoder
from models.models import get_model
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


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
    lin_vals = torch.linspace(border, 1-border, steps=n_vals)
    icdf_vals = torch.cartesian_prod(*([lin_vals]*latent_dims))
    distr = torch.distributions.normal.Normal(torch.zeros(latent_dims), torch.ones(latent_dims))
    z_vals = distr.icdf(icdf_vals)

    return z_vals

def plot_latent_space(model, test_loader):
    n_latent_dims = model.encoder.latent_dim
    if n_latent_dims == 2:
        plot_latent_space_2d(model, test_loader)
        return
    arr = np.zeros((len(test_loader.dataset), n_latent_dims))
    labels = np.zeros((len(test_loader.dataset),))
    n_added = 0
    for image_batch, image_labels in test_loader:
        image_batch = util.dequantize(image_batch)
        image_batch = image_batch.to(device)
        mu, _ = model.encode(image_batch)
        arr[n_added:n_added+len(image_batch), :] = mu.detach().numpy()
        labels[n_added:n_added+len(image_batch)] = image_labels
        n_added += len(image_batch)


    embedded = TSNE(learning_rate='auto', init='pca', perplexity=10).fit_transform(arr)#TSNE(learning_rate='auto', perplexity=50, n_iter=2000, init='pca').fit_transform(arr)

    plt.figure()
    plt.style.use('seaborn')
    scat = plt.scatter(embedded[:, 0], embedded[:, 1], s=10, c=labels, cmap=plt.get_cmap('tab10'))
    cb = plt.colorbar(scat, spacing='proportional')
    plt.show()

def plot_latent_space_2d(model: AutoEncoder, test_loader):
    plt.figure()
    arr = np.zeros([len(test_loader.dataset), 3])
    n_added = 0
    for image_batch, image_labels in test_loader:
        image_batch = util.dequantize(image_batch)
        image_batch = image_batch.to(device)
        mu, _ = model.encode(image_batch)
        arr[n_added:n_added + len(image_batch), :2] = mu.detach().numpy()
        arr[n_added:n_added + len(image_batch), 2] = image_labels
        n_added += len(image_batch)



    plt.style.use('seaborn')
    scat = plt.scatter(arr[:, 0], arr[:, 1], s=10, c=arr[:, 2], cmap=plt.get_cmap('tab10'))
    cb = plt.colorbar(scat, spacing='proportional')
    plt.show()



from util import load_best_model

run = wandb.init()
project_name = 'phase1'
model_name = 'nae'
experiment_name = 'nae_mnist_run_4_latent_size_8_decoder_independent_corner'
latent_dims = 8
image_dim = [1, 28, 28]
alpha = 1e-6
use_center_pixels = False
device = 'cpu'
decoder = 'independent'
dataset = 'mnist'

model = load_best_model(run, project_name, model_name, experiment_name, device, latent_dims, image_dim, alpha, decoder,
                        use_center_pixels, version='best:latest')
test_loader = datasets.get_test_dataloader(dataset)
plot_latent_space(model, test_loader)
# z_vals = get_z_values(n_vals=10, latent_dims=2)
#
# output = model.sample(z=z_vals).detach()
#
# util.plot_image_grid(output, 10, 10, 1)
# plt.show()
# run.finish()