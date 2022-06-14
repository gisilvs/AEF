import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

import wandb
from datasets import get_train_val_dataloaders, get_test_dataloader
from models.model_database import get_model, get_model_denoising

from util import make_averager, dequantize, vae_log_prob, plot_image_grid, bits_per_pixel, count_parameters
from visualize import plot_reconstructions, plot_noisy_reconstructions

parser = argparse.ArgumentParser(description='NAE Experiments')


parser.add_argument('--model', type=str, help='ae | nae-center | nae-corner | nae-external | vae | iwae | vae-iaf | vae-iaf-maf')
parser.add_argument('--dataset', type=str, help='mnist | kmnist | fashionmnist | cifar10')
parser.add_argument('--latent-dims', type=int, help='size of the latent space')
parser.add_argument('--noise-level', type=float, default=0.25,
                    help='amount of noise to add (std of gaussian noise is sampled from) (default = 0.25)')
parser.add_argument('--runs', type=str, default="0", help='run numbers in string format, e.g. "0,1,2,3"')
parser.add_argument('--iterations', type=int, default=100000, help='amount of iterations to train (default: 100,000)')
parser.add_argument('--val-iters', type=int, default=500, help='validate every x iterations (default: 500')
parser.add_argument('--save-iters', type=int, default=2000,
                    help='save model to wandb every x iterations (default: 2,000)')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate (default: 1e-3)')
parser.add_argument('--seed', type=int, default=3, help='seed for the training data (default: 3)')
parser.add_argument('--decoder', type=str, default='fixed',
                    help='fixed (var = 1) | independent (var = s) | dependent (var = s(x))')
parser.add_argument('--custom-name', type=str, help='custom name for wandb tracking')
parser.add_argument('--batch-size', type=int, default=128,
                    help='input batch size for training and testing (default: 128)')

args = parser.parse_args()

assert args.model in ['ae', 'nae-center', 'nae-corner', 'vae', 'iwae', 'vae-iaf', 'nae-external', 'vae-iaf-maf', 'vae-iaf-maf-independent'] # 'maf',
assert args.dataset in ['mnist', 'kmnist', 'emnist', 'fashionmnist', 'cifar10', 'cifar']
assert args.decoder in ['fixed', 'independent', 'dependent']

if args.dataset == 'cifar10':
    args.dataset = 'cifar'

model_name = args.model
decoder = args.decoder
n_iterations = args.iterations
dataset = args.dataset
latent_dims = args.latent_dims
batch_size = args.batch_size
learning_rate = args.lr
use_gpu = True
validate_every_n_iterations = args.val_iters
save_every_n_iterations = args.save_iters
noise_level = args.noise_level

args.runs = [int(item) for item in args.runs.split(',')]

AE_like_models = ['nae-center', 'nae-corner', 'nae-external', 'vae', 'iwae', 'vae-iaf', 'ae', 'vae-iaf-maf', 'vae-iaf-maf-independent']
flow_like_models = ['nae-center', 'nae-corner', 'nae-external', 'maf']

for run_nr in args.runs:
    if args.custom_name is not None:
        run_name = args.custom_name
    else:
        latent_size_str = f"_latent_size_{args.latent_dims}" if model_name in AE_like_models else ""
        decoder_str = f"_decoder_{args.decoder}" if model_name in AE_like_models else ""
        noise_level_str = f"_noise_{noise_level}"
        run_name = f'{args.model}_{args.dataset}_run_{run_nr}{decoder_str}{noise_level_str}'

    config = {
        "model": model_name,
        "dataset": dataset,
        "latent_dims": latent_dims,
        "decoder": args.decoder,
        "learning_rate": learning_rate,
        "n_iterations": n_iterations,
        "batch_size": batch_size,
        "seed": args.seed,
        "noise_level": noise_level
    }

    device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")

    p_validation = 0.1
    train_dataloader, validation_dataloader, image_dim, alpha = get_train_val_dataloaders(dataset, batch_size,
                                                                                          p_validation, seed=args.seed)
    test_dataloader = get_test_dataloader(dataset, batch_size)
    reconstruction_dataloader = get_test_dataloader(dataset, batch_size, shuffle=True)

    n_pixels = np.prod(image_dim)

    model = get_model_denoising(model_name=model_name, decoder=args.decoder,
                                latent_dims=latent_dims, img_shape=image_dim, alpha=alpha)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

    model = model.to(device)

    if not os.path.isdir('./checkpoints'):
        os.mkdir('./checkpoints')

    run = wandb.init(project="denoising-experiments-1", entity="nae",
                     name=run_name, config=config)
    wandb.summary['n_parameters'] = count_parameters(model)
    print('Training ...')

    stop = False
    n_iterations_done = 0
    n_times_validated = 0
    iteration_losses = np.zeros((n_iterations,))
    validation_losses = []
    validation_iterations = []
    validation_reconstruction_errors = []


    for it in range(n_iterations):
        torch.seed()  # Random seed since we fix it at test time
        noise_distribution = torch.distributions.normal.Normal(torch.zeros(batch_size, *image_dim),
                                                               noise_level * torch.ones(batch_size, *image_dim))

        while not stop:
            for training_batch, _ in train_dataloader:
                training_batch = dequantize(training_batch)
                training_batch_noisy = torch.clone(training_batch).detach()
                training_batch_noisy += noise_distribution.sample()[:training_batch.shape[0]]
                training_batch_noisy = torch.clamp(training_batch_noisy, 0., 1.)
                training_batch_noisy = training_batch_noisy.to(device)
                training_batch = training_batch.to(device)

                if model_name in flow_like_models:
                    loss = torch.mean(model.loss_function(training_batch_noisy))
                else:
                    loss = torch.mean(model.loss_function(training_batch_noisy, training_batch_noisy))
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

                        # Create sample plot
                        if model_name != 'ae':
                            samples = model.sample(16)
                            samples = samples.cpu().detach()
                            if model_name == 'maf':
                                samples = samples.view(-1, image_dim[0], image_dim[1], image_dim[2])
                            # plot_image_grid(samples, cols=4)
                            sample_fig = plot_image_grid(samples, cols=4)
                            image_dict = {'samples': sample_fig}
                        else:
                            image_dict = {}

                        # Create reconstruction plots
                        training_reconstruction_fig = plot_noisy_reconstructions(model, reconstruction_dataloader,
                                                                                 device,
                                                                                 noise_distribution, image_dim,
                                                                                 n_rows=6,
                                                                                 image_batch=training_batch)
                        image_dict = {'training_reconstructions': training_reconstruction_fig}

                        reconstruction_fig = plot_noisy_reconstructions(model, reconstruction_dataloader, device,
                                                                            noise_distribution, image_dim, n_rows=6)
                        reconstruction_dict = {'test_reconstructions': reconstruction_fig}

                        # Validation loss and reconstruction error, in both cases using noisy images
                        val_loss_averager = make_averager()
                        val_reconstruction_averager = make_averager()
                        for validation_batch, _ in validation_dataloader:
                            validation_batch = dequantize(validation_batch)
                            validation_batch_noisy = torch.clone(validation_batch).detach()
                            validation_batch_noisy += noise_distribution.sample()[:validation_batch.shape[0]]
                            validation_batch_noisy = torch.clamp(validation_batch_noisy, 0., 1.)
                            validation_batch_noisy = validation_batch_noisy.to(device)

                            if model_name in flow_like_models:
                                loss = torch.mean(model.loss_function(validation_batch_noisy))
                            else:
                                loss = torch.mean(model.loss_function(validation_batch_noisy, validation_batch_noisy))

                            val_loss_averager(loss.item())

                            z = model.encode(validation_batch_noisy)
                            if isinstance(z, tuple):
                                z = z[0]

                            validation_batch_reconstructed = model.decode(z)
                            if isinstance(validation_batch_reconstructed, tuple):
                                validation_batch_reconstructed = validation_batch_reconstructed[0]

                            rce = torch.mean(
                                F.mse_loss(validation_batch_reconstructed, validation_batch_noisy, reduction='none'))

                            val_reconstruction_averager(rce.item())

                        validation_losses.append(val_loss_averager(None))
                        validation_reconstruction_errors.append(val_reconstruction_averager(None))
                        val_metrics = {
                            'val_loss': validation_losses[-1],
                            'val_rce': validation_reconstruction_errors[-1]
                        }


                        if n_iterations_done == 0:
                            best_loss = validation_losses[-1]
                            best_it = n_iterations_done
                            torch.save(model.state_dict(), f'checkpoints/{run_name}_best.pt')
                        # For VAE-like models we save based on validation likelihood
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

                        wandb.log({**metrics, **val_metrics, **image_dict, **histograms, **reconstruction_dict})
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

    artifact_latest = wandb.Artifact(f'{run_name}_latest', type='model')
    artifact_latest.add_file(f'checkpoints/{run_name}_latest.pt')
    run.log_artifact(artifact_latest)
    artifact_best = wandb.Artifact(f'{run_name}_best', type='model')
    artifact_best.add_file(f'checkpoints/{run_name}_best.pt')
    run.log_artifact(artifact_best)

    model.load_state_dict(torch.load(f'checkpoints/{run_name}_best.pt'))
    model.eval()
    test_loss_averager = make_averager()
    test_rce_with_noise_averager = make_averager()
    test_rce_without_noise_averager = make_averager()
    torch.manual_seed(3)  # Seed noise for equal test comparison
    with torch.no_grad():
        # Test loss and RCE with/without noise (comparing to original)
        for test_batch, _ in test_dataloader:
            test_batch = dequantize(test_batch)
            test_batch_noisy = torch.clone(test_batch).detach()
            test_batch_noisy += noise_distribution.sample()[:test_batch.shape[0]]
            test_batch_noisy = torch.clamp(test_batch_noisy, 0., 1.)
            test_batch_noisy = test_batch_noisy.to(device)
            test_batch = test_batch.to(device)

            # Test loss
            if model_name == 'maf':
                test_batch = test_batch.view(-1, torch.prod(torch.tensor(image_dim)))
            if model_name in flow_like_models:
                loss = torch.mean(model.loss_function(test_batch))
            else:
                loss = torch.mean(model.loss_function(test_batch, test_batch))

            test_loss_averager(loss.item())

            # RCE with noise (comparing to original)
            z = model.encode(test_batch_noisy)
            if isinstance(z, tuple):
                z = z[0]

            test_batch_reconstructed = model.decode(z)
            if isinstance(test_batch_reconstructed, tuple):
                test_batch_reconstructed = test_batch_reconstructed[0]

            rce_with_noise = torch.mean(F.mse_loss(test_batch_reconstructed, test_batch, reduction='none'))
            test_rce_with_noise_averager(rce_with_noise.item())

            # RCE without noise (comparing to original)
            z = model.encode(test_batch)
            if isinstance(z, tuple):
                z = z[0]

            test_batch_reconstructed = model.decode(z)
            if isinstance(test_batch_reconstructed, tuple):
                test_batch_reconstructed = test_batch_reconstructed[0]

            rce_without_noise = torch.mean(F.mse_loss(test_batch_reconstructed, test_batch, reduction='none'))
            test_rce_without_noise_averager(rce_without_noise.item())


        test_loss = test_loss_averager(None)
        test_rce_with_noise = test_rce_with_noise_averager(None)
        test_rce_without_noise = test_rce_without_noise_averager(None)
        wandb.summary['test_loss'] = test_loss
        wandb.summary['test_rce_without_noise'] = test_rce_without_noise
        wandb.summary['test_rce_with_noise'] = test_rce_with_noise


        # Approximate log likelihood if model in VAE family
        if model_name in ['vae', 'iwae', 'vae-iaf', 'vae-iaf-maf', 'vae-iaf-maf-independent']:
            test_ll_averager = make_averager()
            for test_batch, _ in test_dataloader:
                test_batch = dequantize(test_batch)
                test_batch = test_batch.to(device)
                for iw_iter in range(20):
                    log_likelihood = torch.mean(model.approximate_marginal(test_batch, n_samples=128))
                    test_ll_averager(log_likelihood.item())
            test_ll = test_ll_averager(None)
            wandb.summary['test_log_likelihood'] = test_ll
            bpp_test = bits_per_pixel(test_ll, n_pixels)
            bpp_test_adjusted = bits_per_pixel(test_ll, n_pixels, adjust_value=256.)

        else:
            bpp_test = bits_per_pixel(test_loss, n_pixels)
            bpp_test_adjusted = bits_per_pixel(test_loss, n_pixels, adjust_value=256.)

        wandb.summary['test_bpp'] = bpp_test
        wandb.summary['test_bpp_adjusted'] = bpp_test_adjusted

        if model_name != 'ae':
            for i in range(5):
                samples = model.sample(16)
                samples = samples.cpu().detach()
                if model_name == 'maf':
                    samples = samples.view(-1, image_dim[0], image_dim[1], image_dim[2])
                fig = plot_image_grid(samples, cols=4)
                image_dict = {'final_samples': fig}
                run.log(image_dict)
        for i in range(5):
            reconstruction_fig = plot_noisy_reconstructions(model, reconstruction_dataloader, device,
                                                            noise_distribution, image_dim, n_rows=9)
            reconstruction_dict = {'final_noisy_reconstructions_test': reconstruction_fig}
            run.log(reconstruction_dict)
        for i in range(5):
            reconstruction_fig = plot_reconstructions(model, test_dataloader, device, image_dim, n_rows=10)
            reconstruction_dict = {'final_reconstructions_test': reconstruction_fig}
            run.log(reconstruction_dict)

    artifact_best = wandb.Artifact(f'{run_name}_best', type='model')
    artifact_best.add_file(f'checkpoints/{run_name}_best.pt')
    run.log_artifact(artifact_best)
    artifact_latest = wandb.Artifact(f'{run_name}_latest', type='model')
    artifact_latest.add_file(f'checkpoints/{run_name}_latest.pt')
    run.log_artifact(artifact_latest)
    wandb.summary['best_iteration'] = best_it


    run.finish()

    plt.close("all")

    # Clean up older artifacts
    api = wandb.Api(overrides={"project": 'denoising-experiments-1', "entity": "nae"})
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
