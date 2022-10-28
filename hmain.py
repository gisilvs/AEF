import argparse
import os
import traceback

import numpy as np
import torch

import wandb
from datasets import get_train_val_dataloaders, get_test_dataloader
from models.hvae import HVAE, HAEF

from metrics import InceptionV3, calculate_fid
from utils.util import make_averager, dequantize, plot_image_grid, bits_per_pixel, count_parameters, \
    has_importance_sampling, get_random_id, get_posterior_scale_aef_linear
from visualize import plot_hierarchical_reconstructions

parser = argparse.ArgumentParser(description='AEF Experiments')
parser.add_argument('--wandb-entity', type=str, help='wandb entity')
parser.add_argument('--wandb-project', type=str, help='wandb project')
parser.add_argument('--model', type=str, help='aef-center | aef-corner | aef-linear | vae')
parser.add_argument('--dataset', type=str, help='mnist | kmnist | fashionmnist | cifar10 | celebahq | celebahq64 | '
                                                'imagenet')
parser.add_argument('--width', type=int, default=64)
parser.add_argument('--custom_width_str', type=str, default="")
parser.add_argument('--encoder_blocks', type=str, default="64x4,64d2,32x4,32d2,16x4,16d2,8x4,8d2,4x4,4d4,1x4")
parser.add_argument('--decoder_blocks', type=str, default="1x4,4m1,4x4,8m4,8x4,16m8,16x4,32m16,32x4,64m32,64x4")
parser.add_argument('--z_dim', type=str, default="{1:64, 4:4, 8:2}")
parser.add_argument("--gaussian_nll", type=int, default=0)
parser.add_argument("--one_stochastic_per_dim", type=int, default=0)
parser.add_argument('--runs', type=int, default=1, help='number of runs to exceute')
parser.add_argument('--iterations', type=int, default=100000, help='amount of iterations to train (default: 100,000)')
parser.add_argument('--val-iters', type=int, default=500, help='validate every x iterations (default: 500')
parser.add_argument('--upload-iters', type=int, default=2000,
                    help='upload model to wandb every x iterations (default: 2,000)')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate (default: 1e-3)')
parser.add_argument('--seed', type=int, default=3, help='seed for the training data (default: 3)')
parser.add_argument('--custom-name', type=str, help='custom name for wandb tracking')
parser.add_argument('--batch-size', type=int, default=128,
                    help='input batch size for training and testing (default: 128)')
parser.add_argument('--data-dir', type=str, default="")

parser.add_argument('--early-stopping', type=int, default=1000000)
parser.add_argument('--gpus', type=str, default="0", help="which gpu(s) to use (default: 0)")

args = parser.parse_args()

model_name = args.model
width = args.width
custom_width_str = args.custom_width_str
encoder_blocks = args.encoder_blocks
decoder_blocks = args.decoder_blocks
z_dim = args.z_dim
gaussian_nll = args.gaussian_nll > 0
one_stochastic_per_dim = args.one_stochastic_per_dim > 0
n_iterations = args.iterations
dataset = args.dataset
batch_size = args.batch_size
learning_rate = args.lr
use_gpu = True
validate_every_n_iterations = args.val_iters
upload_every_n_iterations = args.upload_iters
early_stopping_threshold = args.early_stopping
gpu_nrs = args.gpus

os.environ["CUDA_VISIBLE_DEVICES"] = gpu_nrs
gpu_nr = gpu_nrs[0]

device = torch.device(f"cuda:{gpu_nr}" if use_gpu and torch.cuda.is_available() else "cpu")

for run_nr in range(args.runs):
    p_validation = 0.1
    if dataset == 'imagenet':
        p_validation = 0.01
    train_dataloader, validation_dataloader, image_dim, alpha = get_train_val_dataloaders(dataset, batch_size,
                                                                                          p_validation, seed=args.seed,
                                                                                          data_dir=args.data_dir)
    reconstruction_dataloader = get_test_dataloader(dataset, batch_size, shuffle=True, data_dir=args.data_dir)
    test_dataloader = get_test_dataloader(dataset, batch_size, data_dir=args.data_dir)
    n_pixels = np.prod(image_dim)
    if model_name == 'hvae':
        model = HVAE(width=width,
                     image_size=image_dim[-1],
                     image_channels=image_dim[0],
                     custom_width_str=custom_width_str,
                     encoder_blocks=encoder_blocks,
                     decoder_blocks=decoder_blocks,
                     z_dim=z_dim,
                     gaussian_nll=gaussian_nll,
                     one_stochastic_per_dim=one_stochastic_per_dim)
    elif model_name == 'haef':
        model = HAEF(width=width,
                     image_size=image_dim[-1],
                     image_channels=image_dim[0],
                     custom_width_str=custom_width_str,
                     encoder_blocks=encoder_blocks,
                     decoder_blocks=decoder_blocks,
                     z_dim=z_dim,
                     gaussian_nll=gaussian_nll,
                     one_stochastic_per_dim=one_stochastic_per_dim)
    model = model.to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

    if not os.path.isdir('./checkpoints'):
        os.mkdir('./checkpoints')

    config = {
        "model": model_name,
        "dataset": dataset,
        "z_dim": z_dim,
        "width": width,
        "custom_width_str": custom_width_str,
        "encoder_blocks": encoder_blocks,
        "decoder_blocks": decoder_blocks,
        "gaussian_nll": gaussian_nll,
        "one_stochastic_per_dim": one_stochastic_per_dim,
        "learning_rate": learning_rate,
        "n_iterations": n_iterations,
        "batch_size": batch_size,
        "seed": args.seed,
        "early_stopping": early_stopping_threshold,
        "validate_every_n_iterations": validate_every_n_iterations
    }


    run_name = args.custom_name

    run = wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                     name=run_name, config=config)
    wandb.summary['n_parameters'] = count_parameters(model)

    stop = False
    n_iterations_done = 0
    iteration_losses = np.zeros((n_iterations,))
    validation_losses = []
    n_iterations_without_improvements = 0

    model.train()
    print(f'[Run {run_nr}] Training...')
    for it in range(n_iterations_done, n_iterations):
        while not stop:
            for train_batch, _ in train_dataloader:
                train_batch = train_batch.to(device)

                if model_name == 'maf':
                    train_batch = train_batch.view(-1, torch.prod(torch.tensor(image_dim)))

                train_batch_loss = model.forward(train_batch)
                iteration_losses[n_iterations_done] = train_batch_loss.item()
                log_dictionary = {
                    'train_loss': train_batch_loss,
                }

                optimizer.zero_grad()
                train_batch_loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=200.)
                #if grad_norm < 400 or n_iterations_done < 100:
                optimizer.step()
                #else:
                #    continue

                # We validate first iteration, every n iterations, and at the last iteration
                if (n_iterations_done % validate_every_n_iterations) == 0 or (n_iterations_done == n_iterations - 1):
                    model.eval()

                    with torch.no_grad():
                        samples = model.forward_uncond_samples(16)
                        samples = samples.cpu().detach()
                        if model_name == 'maf':
                            samples = samples.view(-1, image_dim[0], image_dim[1], image_dim[2])

                        sample_img = wandb.Image(plot_image_grid(samples, cols=4))
                        log_dictionary['samples'] = sample_img


                        if model_name != 'maf':
                            reconstruction_img = wandb.Image(plot_hierarchical_reconstructions(model, reconstruction_dataloader, device,
                                                                                  image_dim, n_rows=4))
                            log_dictionary['reconstructions'] = reconstruction_img

                        val_loss_averager = make_averager()
                        for validation_batch, _ in validation_dataloader:
                            validation_batch = validation_batch.to(device)
                            if model_name == 'maf':
                                validation_batch = validation_batch.view(-1, torch.prod(torch.tensor(image_dim)))
                            val_batch_loss = model.forward(validation_batch)
                            val_loss_averager(val_batch_loss.item())

                        validation_losses.append(val_loss_averager(None))
                        log_dictionary['val_loss'] = validation_losses[-1]

                        if n_iterations_done == 0:
                            best_loss = validation_losses[-1]
                            best_it = n_iterations_done
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

    with torch.no_grad():

        # Log samples from best model to wandb
        for i in range(5):
            samples = model.forward_uncond_samples(16)
            samples = samples.cpu().detach()
            if model_name == 'maf':
                samples = samples.view(-1, image_dim[0], image_dim[1], image_dim[2])
            img = plot_image_grid(samples, cols=4)
            image_dict = {'final_samples': wandb.Image(img)}
            run.log(image_dict)
        print(f'[Run {run_nr}] Calculating test loss...')
        test_loss_averager = make_averager()
        for test_batch, _ in test_dataloader:
            test_batch = test_batch.to(device)
            if model_name == 'maf':
                test_batch = test_batch.view(-1, torch.prod(torch.tensor(image_dim)))
            test_batch_loss = model.forward(test_batch)
            test_loss_averager(test_batch_loss.item())
        test_loss = test_loss_averager(None)
        wandb.summary['test_loss'] = test_loss

        if not has_importance_sampling(model):
            bpp_test = bits_per_pixel(test_loss, n_pixels)
            bpp_test_adjusted = bits_per_pixel(test_loss, n_pixels, adjust_value=256.)
            wandb.summary['test_bpp'] = bpp_test
            wandb.summary['test_bpp_adjusted'] = bpp_test_adjusted


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
