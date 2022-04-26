import torch.distributions
import torchvision

import datasets
import util
import wandb
from models.models import get_model
import matplotlib.pyplot as plt


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

from util import load_best_model

run = wandb.init()
project_name = 'prototyping'
model_name = 'nae'
experiment_name = 'vae_mnist_run_0_latent_size_2_decoder_fixed'
latent_dims = 2
image_dim = [1, 28, 28]
alpha = 1e-6
use_center_pixels = True
device = 'cpu'
decoder = 'fixed'

model = load_best_model(run, project_name, model_name, experiment_name, device, latent_dims, image_dim, alpha, decoder,
                        use_center_pixels, version='best:v14')
z_vals = get_z_values(n_vals=10, latent_dims=2)

output = model.sample(z=z_vals).detach()

util.plot_image_grid(output, 10, 10, 1)
plt.show()
run.finish()