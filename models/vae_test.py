
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from nflows.transforms import InverseTransform, AffineTransform

import util

import wandb

from bijectors.actnorm import ActNorm
from bijectors.masked_autoregressive_transform import get_masked_autoregressive_transform
from bijectors.sigmoid import Sigmoid
from datasets import get_train_val_dataloaders, get_test_dataloader
from models.autoencoder import FixedVarianceDecoderSmall
from models.nae_internal import InternalLatentAutoEncoder
from models.vae import VAE
from models.vae_iaf import VAEIAF
from models.vdvae import FixedVarianceDecoderBig, ConvolutionalEncoderBig, IndependentVarianceDecoderBig

from util import make_averager, dequantize, vae_log_prob, plot_image_grid, bits_per_pixel

model_name = 'vdnae-center'
dataset = 'cifar'
latent_dims = 64
batch_size = 64
learning_rate = 5*1e-4
use_gpu = True
n_iterations = 2000
validate_every_n_iterations = 250
save_every_n_iterations = 250
decoder = 'fixed'

run_name = 'test_run_vdnae0'

AE_like_models = ['nae-center', 'nae-corner', 'nae-external', 'vae', 'iwae', 'vae-iaf', 'vdvae', 'vdvae']

config = {
    "model": model_name,
    "dataset": dataset,
    "latent_dims": latent_dims,
    "decoder": decoder,
    "learning_rate": learning_rate,
    "n_iterations": n_iterations,
    "batch_size": batch_size,

}
run = wandb.init(project='prototyping', entity="nae",
                 name=run_name, config=config)

device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")

p_validation = 0.1
train_dataloader, validation_dataloader, image_dim, alpha = get_train_val_dataloaders(dataset, batch_size,
                                                                                      p_validation)
test_dataloader = get_test_dataloader(dataset, batch_size)
n_pixels = np.prod(image_dim)

vae_channels = 64
# decoder = FixedVarianceDecoderBig(output_shape=image_dim,
#                                     latent_ndims=latent_dims)
decoder = IndependentVarianceDecoderBig(output_shape=image_dim,
                                    latent_ndims=latent_dims)

encoder = ConvolutionalEncoderBig(input_shape=image_dim, latent_ndims=latent_dims)
#model = VAE(encoder, decoder)
preprocessing_layers = [InverseTransform(AffineTransform(alpha, 1 - 2 * alpha)), Sigmoid(), ActNorm(image_dim[0])]

core_flow_fn = get_masked_autoregressive_transform
core_flow_pre = core_flow_fn(features=latent_dims,
                             hidden_features=512,
                             num_layers=8,
                             num_blocks_per_layer=2,
                             act_norm_between_layers=True)
core_flow_post = core_flow_fn(features=latent_dims,
                              hidden_features=512,
                              num_layers=8,
                              num_blocks_per_layer=2,
                              act_norm_between_layers=True)

model = InternalLatentAutoEncoder(encoder=encoder, decoder=decoder, core_flow_pre=core_flow_pre,
                                  core_flow_post=core_flow_post, preprocessing_layers=preprocessing_layers,
                                  center_mask=True)
optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

model = model.to(device)

if not os.path.isdir('./checkpoints'):
    os.mkdir('./checkpoints')

print('Training ...')

stop = False
n_iterations_done = 0
n_times_validated = 0
iteration_losses = np.zeros((n_iterations,))
validation_losses = []
validation_iterations = []

for it in range(n_iterations):
    while not stop:
        for image_batch, _ in train_dataloader:
            image_batch = dequantize(image_batch)
            image_batch = image_batch.to(device)

            if model_name == 'maf':
                image_batch = image_batch.view(-1, torch.prod(torch.tensor(image_dim)))

            loss = torch.mean(model.loss_function(image_batch))
            iteration_losses[n_iterations_done] = loss.item()
            metrics = {
                'train_loss': loss
            }

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            # We validate first iteration, every n iterations, and at the last iteration
            if (n_iterations_done % validate_every_n_iterations) == 0 or (n_iterations_done == n_iterations - 1):
                model.eval()

                with torch.no_grad():
                    val_loss_averager = make_averager()

                    samples = model.sample(16)
                    samples = samples.cpu().detach()
                    if model_name == 'maf':
                        samples = samples.view(-1, image_dim[0], image_dim[1], image_dim[2])
                    plot_image_grid(samples, cols=4)
                    image_dict = {'samples': plt}

                    for validation_batch, _ in validation_dataloader:
                        validation_batch = dequantize(validation_batch)
                        validation_batch = validation_batch.to(device)
                        if model_name == 'maf':
                            validation_batch = validation_batch.view(-1, torch.prod(torch.tensor(image_dim)))
                        loss = torch.mean(model.loss_function(validation_batch))
                        val_loss_averager(loss.item())

                    validation_losses.append(val_loss_averager(None))
                    val_metrics = {
                        'val_loss': validation_losses[-1]
                    }
                    if n_iterations_done == 0:
                        best_loss = validation_losses[-1]
                        best_it = n_iterations_done
                    elif validation_losses[-1] < best_loss:
                        best_loss = validation_losses[-1]
                        torch.save(model.state_dict(), f'checkpoints/{run_name}_best.pt')
                        best_it = n_iterations_done
                    validation_iterations.append(n_iterations_done)
                    torch.save({
                        'n_iterations_done': n_iterations_done,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'iteration_losses': iteration_losses,
                        'validation_losses': validation_losses,
                        'best_loss': best_loss},
                        f'checkpoints/{run_name}_latest.pt')
                    n_times_validated += 1
                    histograms = {}
                    for tag, value in model.named_parameters():
                        tag = tag.replace('/', '.')
                        histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                        histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

                    wandb.log({**metrics, **val_metrics, **image_dict, **histograms})
                    plt.close("all")
            else:
                wandb.log(metrics)

            if (n_times_validated > 1) and (n_iterations_done % save_every_n_iterations == 0):
                artifact_latest = wandb.Artifact(f'{run_name}_latest', type='model')
                artifact_latest.add_file(f'checkpoints/{run_name}_latest.pt')
                run.log_artifact(artifact_latest)
                artifact_best = wandb.Artifact(f'{run_name}_best', type='model')
                artifact_best.add_file(f'checkpoints/{run_name}_best.pt')
                run.log_artifact(artifact_best)

            n_iterations_done += 1
            model.train()
            if n_iterations_done >= n_iterations:
                stop = True
                break

model.load_state_dict(torch.load(f'checkpoints/{run_name}_best.pt'))
model.eval()
test_loss_averager = make_averager()
with torch.no_grad():
    for test_batch, _ in test_dataloader:
        test_batch = dequantize(test_batch)
        test_batch = test_batch.to(device)
        if model_name == 'maf':
            test_batch = test_batch.view(-1, torch.prod(torch.tensor(image_dim)))
        loss = torch.mean(model.loss_function(test_batch))
        test_loss_averager(loss.item())
    test_loss = test_loss_averager(None)

    # Approximate log likelihood if model in VAE family
    if model_name in ['vae', 'iwae', 'vae-iaf']:
        test_ll_averager = make_averager()
        for test_batch, _ in test_dataloader:
            test_batch = dequantize(test_batch)
            test_batch = test_batch.to(device)
            for iw_iter in range(20):
                log_likelihood = vae_log_prob(model, test_batch, n_samples=128)
                loss = torch.mean(model.loss_function(test_batch))
                test_ll_averager(loss.item())
        test_ll = test_ll_averager(None)
        wandb.summary['test_log_likelihood'] = test_ll
        bpp_test = bits_per_pixel(test_ll, n_pixels)
        bpp_test_adjusted = bits_per_pixel(test_ll, n_pixels, adjust_value=256.)

    else:
        bpp_test = bits_per_pixel(test_loss, n_pixels)
        bpp_test_adjusted = bits_per_pixel(test_loss, n_pixels, adjust_value=256.)

    wandb.summary['test_bpp'] = bpp_test
    wandb.summary['test_bpp_adjusted'] = bpp_test_adjusted

    for i in range(5):
        samples = model.sample(16)
        samples = samples.cpu().detach()
        if model_name == 'maf':
            samples = samples.view(-1, image_dim[0], image_dim[1], image_dim[2])
        plot_image_grid(samples, cols=4)
        image_dict = {'final_samples': plt}
        run.log(image_dict)

artifact_best = wandb.Artifact(f'{run_name}_best', type='model')
artifact_best.add_file(f'checkpoints/{run_name}_best.pt')
run.log_artifact(artifact_best)
artifact_latest = wandb.Artifact(f'{run_name}_latest', type='model')
artifact_latest.add_file(f'checkpoints/{run_name}_latest.pt')
run.log_artifact(artifact_latest)
wandb.summary['best_iteration'] = best_it
wandb.summary['test_loss'] = test_loss
wandb.summary['n_parameters'] = util.count_parameters(model)

run.finish()

plt.close("all")

# Clean up older artifacts
api = wandb.Api(overrides={"project": "prototyping", "entity": "nae"})
artifact_type, artifact_name = 'model', f'{run_name}_latest'
for version in api.artifact_versions(artifact_type, artifact_name):
    if len(version.aliases) == 0:
        version.delete()
artifact_type, artifact_name = 'model', f'{run_name}_best'
for version in api.artifact_versions(artifact_type, artifact_name):
    if len(version.aliases) == 0:
        version.delete()

# Delete local files if wanted
delete_files_after_upload = False
if delete_files_after_upload:
    os.remove(f'checkpoints/{run_name}_best.pt')
    os.remove(f'checkpoints/{run_name}_latest.pt')

exit()

