import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

import wandb
from datasets import get_train_val_dataloaders, get_test_dataloader
import metrics
from models.model_database import get_model

from util import make_averager, dequantize, plot_image_grid, bits_per_pixel, count_parameters, has_importance_sampling, \
    load_latest_model, get_random_id
from visualize import plot_noisy_reconstructions, plot_noisy_reconstructions

parser = argparse.ArgumentParser(description='AEF Denoising Experiments')
parser.add_argument('--wandb-entity', type=str, default='nae', help='wandb entity')
parser.add_argument('--wandb-project', type=str, default='denoising-experiments-1', help='wandb project')
parser.add_argument('--model', type=str, help='ae | aef-center | aef-corner | aef-linear | vae | iwae')
parser.add_argument('--architecture', type=str, default='small', help='big | small (default)')
parser.add_argument('--posterior-flow', type=str, default='none', help='none (default) | maf | iaf')
parser.add_argument('--prior-flow', type=str, default='none', help='none (default) | maf | iaf')
parser.add_argument('--dataset', type=str, help='mnist | kmnist | fashionmnist | cifar10 | celebahq')
parser.add_argument('--latent-dims', type=int, help='size of the latent space')
parser.add_argument('--noise-level', type=float, default=0.25,
                    help='amount of noise to add (std of gaussian noise is sampled from) (default = 0.25)')
parser.add_argument('--runs', type=int, default=1, help='number of runs to exceute')
parser.add_argument('--iterations', type=int, default=100000, help='amount of iterations to train (default: 100,000)')
parser.add_argument('--val-iters', type=int, default=500, help='validate every x iterations (default: 500')
parser.add_argument('--upload-iters', type=int, default=2000,
                    help='upload latest and best model to wandb every x iterations (default: 2,000)')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate (default: 1e-3)')
parser.add_argument('--seed', type=int, default=3, help='seed for the training data (default: 3)')
parser.add_argument('--decoder', type=str, default='fixed',
                    help='fixed (var = 1) | independent (var = s) | dependent (var = s(x))')
parser.add_argument('--custom-name', type=str, help='custom name for wandb tracking')
parser.add_argument('--batch-size', type=int, default=128,
                    help='input batch size for training and testing (default: 128)')
parser.add_argument('--data-dir', type=str, default="")
parser.add_argument('--gpus', type=str, default="0", help="which gpu(s) to use (default: 0)")
parser.add_argument('--early-stopping', type=int, default=1000000)

args = parser.parse_args()

assert args.model in ['ae', 'aef-center', 'aef-corner', 'aef-linear', 'vae', 'iwae']
assert args.dataset in ['mnist', 'kmnist', 'emnist', 'fashionmnist', 'cifar', 'celebahq']
assert args.decoder in ['fixed', 'independent', 'dependent']
assert args.posterior_flow in ['none', 'maf', 'iaf']
assert args.prior_flow in ['none', 'maf', 'iaf']
assert args.architecture in ['small', 'big']

model_name = args.model
decoder = args.decoder
n_iterations = args.iterations
dataset = args.dataset
latent_dims = args.latent_dims
batch_size = args.batch_size
learning_rate = args.lr
use_gpu = True
validate_every_n_iterations = args.val_iters
upload_every_n_iterations = args.upload_iters
noise_level = args.noise_level
architecture_size = args.architecture
posterior_flow = args.posterior_flow
prior_flow = args.prior_flow
gpu_nrs = args.gpus
early_stopping_threshold = args.early_stopping
data_dir = None if args.data_dir == "" else args.data_dir

os.environ["CUDA_VISIBLE_DEVICES"] = gpu_nrs
gpu_nr = gpu_nrs[0]

device = torch.device(f"cuda:{gpu_nr}" if use_gpu and torch.cuda.is_available() else "cpu")

print(f"Starting {args.runs} runs with the following configuration:")
print(f"Model: {model_name}\nDataset: {dataset}\nLatent dimensions: {latent_dims}\nDecoder: {decoder}\nNoise level: {noise_level}\nLearning rate: {learning_rate}\nNumber of iterations: {n_iterations}\nBatch size: {batch_size}")


for run_nr in range(args.runs):
    p_validation = 0.1
    train_dataloader, validation_dataloader, image_dim, alpha = get_train_val_dataloaders(dataset, batch_size,
                                                                                          p_validation, seed=args.seed,
                                                                                          data_dir=args.data_dir)
    test_dataloader = get_test_dataloader(dataset, batch_size, data_dir=args.data_dir)
    reconstruction_dataloader = get_test_dataloader(dataset, batch_size, shuffle=True, data_dir=args.data_dir)
    n_pixels = np.prod(image_dim)

    model = get_model(model_name=model_name, architecture_size=architecture_size, decoder=args.decoder,
                      latent_dims=latent_dims, img_shape=image_dim, alpha=alpha,
                      posterior_flow_name=posterior_flow, prior_flow_name=prior_flow)
    model = model.to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

    if not os.path.isdir('./checkpoints'):
        os.mkdir('./checkpoints')

    config = {
        "model": model_name,
        "dataset": dataset,
        "latent_dims": latent_dims,
        "decoder": decoder,
        "learning_rate": learning_rate,
        "n_iterations": n_iterations,
        "batch_size": batch_size,
        "seed": args.seed,
        "noise_level": noise_level,
        "architecture_size": architecture_size,
        "posterior_flow": posterior_flow,
        "prior_flow": prior_flow,
        "preprocessing": True,
        "early_stopping": early_stopping_threshold
    }

    if args.custom_name is not None:
        run_name = args.custom_name
    else:
        run_id = get_random_id(4)
        latent_size_str = f"_latent_size_{args.latent_dims}" if model_name != 'MAF' else ""
        decoder_str = f"_decoder_{args.decoder}" if model_name in model_name != 'MAF' else ""
        architecture_str = f"_{architecture_size}" if model_name in model_name != 'MAF' else ""
        post_flow_str = f"_post_{posterior_flow}" if posterior_flow != 'none' else ""
        prior_flow_str = f"_prior_{prior_flow}" if prior_flow != 'none' else ""
        noise_level_str = f"_noise_{noise_level}"
        run_name = f'{args.model}{architecture_str}_{args.dataset}{noise_level_str}{post_flow_str}{prior_flow_str}_{run_id}'


    run = wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                         name=run_name, config=config)
    wandb.summary['n_parameters'] = count_parameters(model)

    stop = False
    n_iterations_done = 0
    iteration_losses = np.zeros((n_iterations,))
    validation_losses = []
    validation_reconstruction_errors = []
    n_iterations_without_improvements = 0

    model.train()
    print(f'[Run {run_nr}] Training...')

    for it in range(n_iterations):
        torch.seed()  # Random seed since we fix it at test time
        noise_distribution = torch.distributions.normal.Normal(torch.zeros(batch_size, *image_dim).to(device),
                                                               noise_level * torch.ones(batch_size, *image_dim).to(
                                                                   device))

        while not stop:
            for training_batch, _ in train_dataloader:
                training_batch = dequantize(training_batch)
                training_batch = training_batch.to(device)
                training_batch_noisy = torch.clone(training_batch).detach().to(device)
                training_batch_noisy += noise_distribution.sample()[:training_batch.shape[0]]
                training_batch_noisy = torch.clamp(training_batch_noisy, 0., 1.)
                training_batch_noisy = training_batch_noisy.to(device)

                train_batch_loss = torch.mean(model.loss_function(training_batch_noisy))

                iteration_losses[n_iterations_done] = train_batch_loss.item()
                log_dictionary = {
                    'train_loss': train_batch_loss
                }

                optimizer.zero_grad()
                train_batch_loss.backward()

                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                           max_norm=200.)

                optimizer.step()

                # We validate first iteration, every n iterations, and at the last iteration
                if (n_iterations_done % validate_every_n_iterations) == 0 or (n_iterations_done == n_iterations - 1):
                    model.eval()
                    with torch.no_grad():
                        # Create sample plot
                        if model_name != 'ae':
                            samples = model.sample(16)
                            samples = samples.cpu().detach()
                            sample_img = wandb.Image(plot_image_grid(samples, cols=4))
                            log_dictionary['samples'] = sample_img

                        # Create reconstruction plot
                        training_reconstruction_img = wandb.Image(plot_noisy_reconstructions(model, training_batch,
                                                                                             device,
                                                                                             noise_distribution, image_dim,
                                                                                             n_rows=6, n_cols=6))
                        log_dictionary['training_reconstructions'] = training_reconstruction_img

                        # Validation loss and reconstruction error, in both cases using noisy images
                        val_loss_averager = make_averager()
                        val_reconstruction_averager = make_averager()
                        for validation_batch, _ in validation_dataloader:
                            validation_batch = dequantize(validation_batch)
                            validation_batch = validation_batch.to(device)

                            validation_batch_noisy = torch.clone(validation_batch).detach().to(device)
                            validation_batch_noisy += noise_distribution.sample()[:validation_batch.shape[0]]
                            validation_batch_noisy = torch.clamp(validation_batch_noisy, 0., 1.)

                            val_batch_loss = torch.mean(model.loss_function(validation_batch_noisy))
                            val_loss_averager(val_batch_loss.item())

                            z = model.encode(validation_batch_noisy)
                            if isinstance(z, tuple):
                                z = z[0]
                            validation_batch_reconstructed = model.decode(z)

                            rce = torch.mean(
                                F.mse_loss(validation_batch_reconstructed, validation_batch_noisy, reduction='none'))
                            val_reconstruction_averager(rce.item())

                        validation_losses.append(val_loss_averager(None))
                        validation_reconstruction_errors.append(val_reconstruction_averager(None))
                        log_dictionary['val_loss'] = validation_losses[-1]
                        log_dictionary['val_rce'] = validation_reconstruction_errors[-1]

                        if n_iterations_done == 0:
                            best_loss = validation_losses[-1]
                            best_it = n_iterations_done
                            torch.save(model.state_dict(), f'checkpoints/{run_name}_best.pt')
                        # We save based on validation loss (in autoencoder models this is validation log-likelihood)
                        elif validation_losses[-1] < best_loss:
                            n_iterations_without_improvements = 0
                            best_loss = validation_losses[-1]
                            torch.save(model.state_dict(), f'checkpoints/{run_name}_best.pt')
                            best_it = n_iterations_done
                        else:
                            n_iterations_without_improvements += validate_every_n_iterations

                        torch.save({
                            'n_iterations_done': n_iterations_done,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'iteration_losses': iteration_losses,
                            'validation_losses': validation_losses,
                            'best_loss': best_loss},
                            f'checkpoints/{run_name}_latest.pt')

                        log_dictionary['iterations_without_improvement'] = n_iterations_without_improvements
                        wandb.log(log_dictionary)


                        if n_iterations_without_improvements >= early_stopping_threshold:
                            stop = True
                            break
                else:
                    wandb.log({'train_loss': train_batch_loss})

                if (n_iterations_done > validate_every_n_iterations) and \
                        ((n_iterations_done % upload_every_n_iterations == 0) or
                         (n_iterations_done + 1) == n_iterations):
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

    # Save latest and best model
    artifact_latest = wandb.Artifact(f'{run_name}_latest', type='model')
    artifact_latest.add_file(f'checkpoints/{run_name}_latest.pt')
    run.log_artifact(artifact_latest)
    artifact_best = wandb.Artifact(f'{run_name}_best', type='model')
    artifact_best.add_file(f'checkpoints/{run_name}_best.pt')
    run.log_artifact(artifact_best)
    wandb.summary['best_iteration'] = best_it

    # We calculate final results on the best model
    model.load_state_dict(torch.load(f'checkpoints/{run_name}_best.pt'))
    model.eval()

    test_loss_averager = make_averager()
    test_rce_with_noise_averager = make_averager()
    test_rce_without_noise_averager = make_averager()
    torch.manual_seed(3)  # Seed noise for equal test comparison
    print(f'[Run {run_nr}] Calculating test loss...')
    with torch.no_grad():
        # Test loss and RCE with/without noise (comparing to original)
        for test_batch, _ in test_dataloader:
            test_batch = dequantize(test_batch)
            test_batch = test_batch.to(device)
            test_batch_noisy = torch.clone(test_batch).detach().to(device)
            test_batch_noisy += noise_distribution.sample()[:test_batch.shape[0]]
            test_batch_noisy = torch.clamp(test_batch_noisy, 0., 1.)
            test_batch_noisy = test_batch_noisy.to(device)

            test_loss = torch.mean(model.loss_function(test_batch))
            test_loss_averager(test_loss.item())

            # RCE with noise (comparing to original)
            z = model.encode(test_batch_noisy)
            if isinstance(z, tuple):
                z = z[0]
            test_batch_reconstructed = model.decode(z)

            rce_with_noise = torch.mean(F.mse_loss(test_batch_reconstructed, test_batch, reduction='none'))
            test_rce_with_noise_averager(rce_with_noise.item())

            # RCE without noise (comparing to original)
            z = model.encode(test_batch)
            if isinstance(z, tuple):
                z = z[0]
            test_batch_reconstructed = model.decode(z)

            rce_without_noise = torch.mean(F.mse_loss(test_batch_reconstructed, test_batch, reduction='none'))
            test_rce_without_noise_averager(rce_without_noise.item())

        test_loss = test_loss_averager(None)
        test_rce_with_noise = test_rce_with_noise_averager(None)
        test_rce_without_noise = test_rce_without_noise_averager(None)
        wandb.summary['test_loss'] = test_loss
        wandb.summary['test_rce_without_noise'] = test_rce_without_noise
        wandb.summary['test_rce_with_noise'] = test_rce_with_noise

        if model_name != 'ae':
            for i in range(5):
                samples = model.sample(16)
                samples = samples.cpu().detach()

                img = wandb.Image(plot_image_grid(samples, cols=4))
                run.log({'final_samples': img})
        test_iterator = iter(test_dataloader)
        for i in range(5):
            test_batch, _ = next(test_iterator)
            test_batch = test_batch.to(device)
            reconstruction_img = wandb.Image(plot_noisy_reconstructions(model, test_batch, device,
                                                                        noise_distribution, image_dim, n_rows=9, n_cols=9))
            run.log({'final_noisy_reconstructions_test': reconstruction_img})

        # Calculate IFE
        incept = metrics.InceptionV3().to(device)
        ife = metrics.calculate_ife(model, dataset, device, noise_distribution, batch_size=batch_size, incept=incept,
                                    data_dir=data_dir)
        wandb.summary['ife'] = ife

    run.finish()

    # Clean up older artifacts
    api = wandb.Api(overrides={"project": args.wandb_project, "entity": args.wandb_entity})
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
