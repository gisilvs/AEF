import torch

import metrics
import wandb
f
from util import load_best_model


run = wandb.init()


## Specify which run you want to select
project_name = 'phase1'
image_dim = [1, 28, 28]

latent_sizes = [2, 4, 8, 16]

alpha = 1e-6
use_center_pixels = False
use_gpu = False
device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")

dataset = 'mnist'
#model_name = 'nae-center'
model_name = 'vae'
latent_size = 8
latent_dims = latent_size

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

model = load_best_model(run, project_name, model_name, experiment_name, device, decoder, latent_dims, image_dim,
                            alpha, use_center_pixels, version='best:latest')

model = model.to(device)
metrics.calculate_fid(model, dataset, device, n_samples=32)
run.finish()
exit()

