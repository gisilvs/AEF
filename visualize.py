import torch.distributions

import datasets
import wandb
from models.models import get_model


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

def get_z_values(n_vals: int = 20, border: float = 0.15, distr: torch.distributions.Distribution, latent_dims: int = 2):
    lin_vals = torch.linspace(border, 1-border, steps=n_vals)
    grid = torch.meshgrid(*[lin_vals]*latent_dims)