import torch
import wandb
from util import load_best_model

run = wandb.init()
project_name = 'test-project'
model_name = 'nae'
experiment_name = 'model_best'#'nae_fashionmnist_latent_size(16)_0_indvar_center'
latent_dims = 16
image_dim = [1,28,28]
alpha = 1e-6
use_center_pixels = True
device = 'cpu'

model = load_best_model(run, project_name, model_name, experiment_name, device, latent_dims, image_dim, alpha, use_center_pixels, version='latest')