import torch

import wandb
from datasets import get_test_dataloader, get_train_val_dataloaders

import numpy as np
from metrics import InceptionV3, get_statistics_numpy, calculate_frechet_distance
from util import load_best_model


run = wandb.init()

project_name = 'phase1'
image_dim = [1, 28, 28]

latent_sizes = [2, 4, 8, 16]

alpha = 1e-6
use_center_pixels = False
use_gpu = False
device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")

latent_grid_size = 20
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
#n_samples = 10000
n_samples = 1024
batch_size = 32

test_loader = get_test_dataloader(dataset, batch_size)

activations_test = np.empty((n_samples, InceptionV3.DEFAULT_DIMS))

incept = InceptionV3().to(device)
incept.eval()
start_idx = 0
for batch, _ in test_loader:
    if batch.shape[1] == 1:
        # HACK: Inception expects three channels so we tile
        batch = batch.repeat((1,3,1,1))

    with torch.no_grad():
        batch_activations = incept(batch)[0].squeeze(3).squeeze(2).cpu().numpy()

    activations_test[start_idx:start_idx+batch_size, :] = batch_activations
    start_idx = start_idx + batch_size
    if start_idx >= n_samples:
        break

train_loader = get_train_val_dataloaders(dataset, batch_size, p_validation=0, return_img_dim=False, return_alpha=False)[0]

activations_train = np.empty((n_samples, InceptionV3.DEFAULT_DIMS))

start_idx = 0
for batch, _ in train_loader:
    if batch.shape[1] == 1:
        # HACK: Inception expects three channels so we tile
        batch = batch.repeat((1,3,1,1))

    with torch.no_grad():
        batch_activations = incept(batch)[0].squeeze(3).squeeze(2).cpu().numpy()

    activations_train[start_idx:start_idx+batch_size, :] = batch_activations
    start_idx = start_idx + batch_size
    if start_idx >= n_samples:
        break

activations_samples = np.empty((n_samples, InceptionV3.DEFAULT_DIMS))
n_filled = 0
min_max_samples = np.empty((n_samples, 2))
while n_filled < n_samples:
    samples = model.sample(batch_size, temperature=1)
    min_max_samples[n_filled:n_filled+batch_size, 0] = torch.min(samples.view(batch_size, -1).cpu().detach(), dim=1).values.numpy()
    min_max_samples[n_filled:n_filled+batch_size, 1] = torch.max(samples.view(batch_size, -1).cpu().detach(), dim=1).values.numpy()
    if samples.shape[1] == 1:
        # HACK: Inception expects three channels so we tile
        samples = samples.repeat((1, 3, 1, 1))

        with torch.no_grad():
            batch_activations = incept(samples)[0].squeeze(3).squeeze(2).cpu().numpy()
        activations_samples[n_filled:n_filled+batch_size] = batch_activations
        n_filled += batch_size

train_mu, train_cov = get_statistics_numpy(activations_train)
test_mu, test_cov = get_statistics_numpy(activations_test)
samples_mu, samples_cov = get_statistics_numpy(activations_samples)

fid_test_sam = calculate_frechet_distance(samples_mu, samples_cov, test_mu, test_cov)
fid_test_train = calculate_frechet_distance(train_mu, train_cov, test_mu, test_cov)
fid_train_sam = calculate_frechet_distance(train_mu, train_cov, samples_mu, samples_cov)
print(f'FID between test and train set: {fid_test_train}')
print(f'FID between test set and generated samples: {fid_test_sam}')
print(f'FID between train set and generated samples: {fid_train_sam}')
print(np.min(min_max_samples), np.max(min_max_samples))
run.finish()
exit()

