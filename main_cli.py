import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

import wandb
from datasets import get_train_val_dataloaders, get_test_dataloader
from models.model_database import get_model

from util import make_averager, dequantize, vae_log_prob, plot_image_grid, bits_per_pixel, count_parameters
from visualize import plot_reconstructions

parser = argparse.ArgumentParser(description='NAE Experiments')
parser.add_argument('--wandb-type', type=str, help='phase1 | phase2 | prototyping | visualization')
parser.add_argument('--model', type=str, help='nae-center | nae-corner | nae-external | vae | iwae | vae-iaf | maf')
parser.add_argument('--architecture', type=str, default='small', help='big | small (default)')
parser.add_argument('--post-flow', type=str, default='none', help='none (default) | maf | iaf')
parser.add_argument('--prior-flow', type=str, default='none', help='none (default) | maf | iaf')
parser.add_argument('--dataset', type=str, help='mnist | kmnist | fashionmnist | cifar10')
parser.add_argument('--latent-dims', type=int, help='size of the latent space')
parser.add_argument('--runs', type=str, help='run numbers in string format, e.g. "0,1,2,3"')
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

assert args.wandb_type in ['phase1', 'phase2', 'prototyping', 'visualization']
assert args.model in ['nae-center', 'nae-corner', 'vae', 'iwae', 'vae-iaf', 'maf', 'nae-external']
assert args.post_flow in ['none', 'maf', 'iaf']
assert args.prior_flow in ['none', 'maf', 'iaf']
assert args.dataset in ['mnist', 'kmnist', 'emnist', 'fashionmnist', 'cifar10', 'cifar']
assert args.decoder in ['fixed', 'independent', 'dependent']
assert args.architecture in ['small', 'big']

if args.architecture == 'big':
    assert args.dataset in ['cifar', 'cifar10']

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
architecture_size = args.architecture
posterior_flow = args.post_flow
prior_flow = args.prior_flow

args.runs = [int(item) for item in args.runs.split(',')]

AE_like_models = ['nae-center', 'nae-corner', 'nae-external', 'vae', 'iwae', 'vae-iaf']

for run_nr in args.runs:
    if args.custom_name is not None:
        run_name = args.custom_name
    else:
        latent_size_str = f"_latent_size_{args.latent_dims}" if model_name in AE_like_models else ""
        decoder_str = f"_decoder_{args.decoder}" if model_name in AE_like_models else ""
        architecture_str = f"_{architecture_size}" if model_name in AE_like_models else ""
        post_flow_str = f"_post_{posterior_flow}" if posterior_flow is not 'none' else ""
        prior_flow_str = f"_prior_{prior_flow}" if prior_flow is not 'none' else ""
        run_name = f'{args.model}{architecture_str}_{args.dataset}_run_{run_nr}{latent_size_str}{decoder_str}{post_flow_str}{prior_flow_str}'

    config = {
        "model": model_name,
        "dataset": dataset,
        "latent_dims": latent_dims,
        "decoder": args.decoder,
        "learning_rate": learning_rate,
        "n_iterations": n_iterations,
        "batch_size": batch_size,
        "seed": args.seed,
        "architecture_size": architecture_size,
        "posterior_flow": posterior_flow,
        "prior_flow": prior_flow,
    }

    device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")

    p_validation = 0.1
    train_dataloader, validation_dataloader, image_dim, alpha = get_train_val_dataloaders(dataset, batch_size,
                                                                                          p_validation, seed=args.seed)
    reconstruction_dataloader = get_test_dataloader(dataset, batch_size, shuffle=True)
    test_dataloader = get_test_dataloader(dataset, batch_size)
    n_pixels = np.prod(image_dim)

    model = get_model(model_name=model_name, architecture_size=architecture_size, decoder=args.decoder,
                      latent_dims=latent_dims, img_shape=image_dim, alpha=alpha,
                      posterior_flow_name=posterior_flow, prior_flow_name=prior_flow)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

    model = model.to(device)

    if not os.path.isdir('./checkpoints'):
        os.mkdir('./checkpoints')

    run = wandb.init(project=args.wandb_type, entity="nae",
                     name=run_name, config=config)

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
                        #plot_image_grid(samples, cols=4)
                        sample_fig = plot_image_grid(samples, cols=4)
                        image_dict = {'samples': sample_fig}

                        if model_name != 'maf':
                            reconstruction_fig = plot_reconstructions(model, reconstruction_dataloader, device,
                                                                      image_dim, n_rows=4)
                            reconstruction_dict = {'reconstructions': reconstruction_fig}

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
            fig = plot_image_grid(samples, cols=4)
            image_dict = {'final_samples': fig}
            run.log(image_dict)

    artifact_best = wandb.Artifact(f'{run_name}_best', type='model')
    artifact_best.add_file(f'checkpoints/{run_name}_best.pt')
    run.log_artifact(artifact_best)
    artifact_latest = wandb.Artifact(f'{run_name}_latest', type='model')
    artifact_latest.add_file(f'checkpoints/{run_name}_latest.pt')
    run.log_artifact(artifact_latest)
    wandb.summary['best_iteration'] = best_it
    wandb.summary['test_loss'] = test_loss
    wandb.summary['n_parameters'] = count_parameters(model)

    run.finish()

    plt.close("all")

    # Clean up older artifacts
    api = wandb.Api(overrides={"project": args.wandb_type, "entity": "nae"})
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
