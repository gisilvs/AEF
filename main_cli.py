import argparse
import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
from nflows.transforms.base import InverseTransform
from bijectors.sigmoid import Sigmoid
from nflows.transforms.standard import AffineTransform

import wandb
from datasets import get_train_val_dataloaders, get_test_dataloader
from bijectors.actnorm import ActNorm
from models.autoencoder import ConvolutionalEncoder, IndependentVarianceDecoder
from models.iwae import IWAE
from models.models import get_model
from models.normalizing_autoencoder import NormalizingAutoEncoder
from models.vae import VAE
from models.vae_iaf import VAEIAF
from util import make_averager, dequantize, vae_log_prob

parser = argparse.ArgumentParser(description='NAE Experiments')
parser.add_argument('--model', type=str, help='nae | vae | iwae | vae-iaf | maf')
parser.add_argument('--dataset', type=str, help='mnist | kmnist | fashionmnist | cifar10')
parser.add_argument('--latent-dims', type=int, help='size of the latent space')
parser.add_argument('--batch-size', type=int, default=128,
                    help='input batch size for training and testing (default: 128)')
parser.add_argument('--runs', type=str, help='run numbers in string format, e.g. "0,1,2,3"')
parser.add_argument('--iterations', type=int, default=100000, help='amount of iterations to train (default: 100,000)')
parser.add_argument('--val-iters', type=int, default=500, help='validate every x iterations (default: 5,000')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate (default: 1e-3)')
parser.add_argument('--seed', type=int, default=3, help='seed for the training data (default: 3)')
parser.add_argument('--use-center', action='store_true',
                    help='add this flag to use center core pixels (default: corner pixels)')

args = parser.parse_args()

assert args.model in ['nae', 'vae', 'iwae', 'vae-iaf', 'maf']
assert args.dataset in ['mnist', 'kmnist', 'emnist', 'fashionmnist', 'cifar10']

model_name = args.model
n_iterations = args.iterations
dataset = args.dataset
latent_dims = args.latent_dims
batch_size = args.batch_size
learning_rate = args.lr
use_gpu = True
validate_every_n_iterations = args.val_iters
use_center_pixels = args.use_center

args.runs = [int(item) for item in args.runs.split(',')]

for model_nr in args.runs:
    use_center_pixels_str = "_center" if use_center_pixels else "_corner"
    use_center_pixels_str = use_center_pixels_str if model_name == 'nae' else ""
    run_name = f'{args.model}_{args.dataset}_latent_size({args.latent_dims})_{model_nr}_indvar{use_center_pixels_str}'
    config = {
        "learning_rate": learning_rate,
        "n_iterations": n_iterations,
        "batch_size": batch_size,
        "dataset": dataset,
        "latent_dims": latent_dims,
        "use_center_pixels": use_center_pixels
    }
    run = wandb.init(project="test-project", entity="nae",
                     name=run_name, config=config)

    device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    p_validation = 0.1
    train_dataloader, validation_dataloader, image_dim, alpha = get_train_val_dataloaders(dataset, batch_size,
                                                                                          p_validation, seed=args.seed)
    test_dataloader = get_test_dataloader(dataset, batch_size)

    model = get_model(model_name, latent_dims, image_dim, alpha, use_center_pixels)
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
    averaging_window_size = 10

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
                    'train/train_loss': loss
                }

                # backpropagation
                optimizer.zero_grad()
                loss.backward()

                # one step of the optmizer
                optimizer.step()

                # We validate first epoch
                if (n_iterations_done % validate_every_n_iterations) == 0 or (n_iterations_done == n_iterations - 1):
                    model.eval()

                    val_loss_averager = make_averager()

                    # todo: move this to utils with plotting function
                    samples = model.sample(16)
                    samples = samples.cpu().detach().numpy()
                    _, axs = plt.subplots(4, 4, )
                    axs = axs.flatten()
                    for img, ax in zip(samples, axs):
                        ax.axis('off')
                        ax.imshow(img.reshape(28, 28), cmap='gray')
                    image_dict = {'samples': plt}

                    with torch.no_grad():
                        for validation_batch, _ in validation_dataloader:
                            validation_batch = dequantize(validation_batch)
                            validation_batch = validation_batch.to(device)
                            if model_name == 'maf':
                                validation_batch = validation_batch.view(-1, torch.prod(torch.tensor(image_dim)))
                            loss = torch.mean(model.loss_function(validation_batch))
                            val_loss_averager(loss.item())

                        validation_losses.append(val_loss_averager(None))
                        val_metrics = {
                            'val/val_loss': validation_losses[-1]
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

                else:
                    wandb.log(metrics)

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
                log_likelihood = vae_log_prob(model, test_batch, n_samples=128)
                loss = torch.mean(model.loss_function(test_batch))
                test_ll_averager(loss.item())
            test_ll = test_ll_averager(None)
            wandb.summary['test_log_likelihood'] = test_ll


        for i in range(5):
            samples = model.sample(16)
            samples = samples.cpu().detach().numpy()
            _, axs = plt.subplots(4, 4, )
            axs = axs.flatten()
            for img, ax in zip(samples, axs):
                ax.axis('off')
                ax.imshow(img.reshape(28, 28), cmap='gray')
            image_dict = {'final_samples': plt}
            run.log(image_dict)

    artifact_best = wandb.Artifact('model_best', type='model')
    artifact_best.add_file(f'checkpoints/{run_name}_best.pt')
    run.log_artifact(artifact_best)
    artifact_latest = wandb.Artifact('model_latest', type='model')
    artifact_latest.add_file(f'checkpoints/{run_name}_latest.pt')
    run.log_artifact(artifact_latest)
    wandb.summary['best_iteration'] = best_it
    wandb.summary['test_loss'] = test_loss

    run.finish()