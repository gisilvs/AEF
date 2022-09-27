import argparse
import os
import traceback

import matplotlib.pyplot as plt
import numpy as np
import torch

from datasets import get_train_val_dataloaders, get_test_dataloader
from models.model_database import get_model
from metrics import InceptionV3, calculate_fid
from util import make_averager, dequantize, plot_image_grid, bits_per_pixel, has_importance_sampling, get_random_id, \
    get_posterior_scale_aef_linear

parser = argparse.ArgumentParser(description='AEF Experiments')
parser.add_argument('--model', type=str, help='aef-center | aef-corner | aef-linear | vae | iwae | vae-iaf | maf')
parser.add_argument('--architecture', type=str, default='small', help='big | small (default)')
parser.add_argument('--posterior-flow', type=str, default='none', help='none (default) | maf | iaf')
parser.add_argument('--prior-flow', type=str, default='none', help='none (default) | maf | iaf')
parser.add_argument('--dataset', type=str, help='mnist | kmnist | fashionmnist | cifar10 | celebahq | celebahq64 | imagenet')
parser.add_argument('--latent-dims', type=int, help='size of the latent space')
parser.add_argument('--iterations', type=int, default=100000, help='amount of iterations to train (default: 100,000)')
parser.add_argument('--val-iters', type=int, default=500, help='validate every x iterations (default: 500')
parser.add_argument('--save-iters', type=int, default=2000,
                    help='save model every x iterations (default: 2,000)')
parser.add_argument('--custom-name', type=str, help='custom name for the model save file')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate (default: 1e-3)')
parser.add_argument('--seed', type=int, default=3, help='seed for the training data (default: 3)')
parser.add_argument('--decoder', type=str, default='independent',
                    help='fixed (var = 1) | independent (var = s) | dependent (var = s(x))')
parser.add_argument('--batch-size', type=int, default=128,
                    help='input batch size for training and testing (default: 128)')
parser.add_argument('--data-dir', type=str, default="")
parser.add_argument('--early-stopping', type=int, default=1000000, help='early stopping parameter: stop training after N iterations without a validation loss decrease')


args = parser.parse_args()

assert args.model in ['aef-center', 'aef-corner', 'aef-linear', 'vae']
assert args.posterior_flow in ['none', 'maf', 'iaf']
assert args.prior_flow in ['none', 'maf', 'iaf']
assert args.dataset in ['mnist', 'kmnist', 'emnist', 'fashionmnist', 'cifar10', 'cifar', 'imagenet', 'celebahq', 'celebahq64']
assert args.decoder in ['fixed', 'independent', 'dependent']
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
save_every_n_iterations = args.save_iters
architecture_size = args.architecture
posterior_flow = args.posterior_flow
prior_flow = args.prior_flow
early_stopping_threshold = args.early_stopping

device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")

print(f"Model: {model_name}\nPosterior flow: {posterior_flow}\nPrior flow: {prior_flow}\nDataset: {dataset}\n"
      f"Latent dimensions: {latent_dims}\nArchitecture size: {architecture_size}\nDecoder: {decoder}\n"
      f"Learning rate: {learning_rate}\nNumber of iterations: {n_iterations}\nBatch size: {batch_size}\n"
      f"Early stopping threshold: {early_stopping_threshold}")

p_validation = 0.1
if dataset == 'imagenet':
    p_validation = 0.01
train_dataloader, validation_dataloader, image_dim, alpha = get_train_val_dataloaders(dataset, batch_size,
                                                                                      p_validation, seed=args.seed, data_dir=args.data_dir)
reconstruction_dataloader = get_test_dataloader(dataset, batch_size, shuffle=True, data_dir=args.data_dir)
test_dataloader = get_test_dataloader(dataset, batch_size, data_dir=args.data_dir)
n_pixels = np.prod(image_dim)

model = get_model(model_name=model_name, architecture_size=architecture_size, decoder=args.decoder,
                  latent_dims=latent_dims, img_shape=image_dim, alpha=alpha,
                  posterior_flow_name=posterior_flow, prior_flow_name=prior_flow)
optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

model = model.to(device)

if args.custom_name is not None:
    run_name = args.custom_name
else:
    run_id = get_random_id(4)
    latent_size_str = f"_latent_size_{args.latent_dims}"
    decoder_str = f"_decoder_{args.decoder}"
    architecture_str = f"_{architecture_size}"
    post_flow_str = f"_post_{posterior_flow}" if posterior_flow != 'none' else ""
    prior_flow_str = f"_prior_{prior_flow}" if prior_flow != 'none' else ""
    run_name = f'{args.model}{architecture_str}_{args.dataset}_{latent_size_str}{decoder_str}{post_flow_str}{prior_flow_str}_{run_id} '

if not os.path.isdir('./checkpoints'):
    os.mkdir('./checkpoints')

print('Training ...')

stop = False
n_iterations_done = 0
iteration_losses = np.zeros((n_iterations,))
validation_losses = []
n_iterations_without_improvements = 0

model.train()
for it in range(n_iterations_done, n_iterations):
    while not stop:
        for train_batch, _ in train_dataloader:
            train_batch = dequantize(train_batch)
            train_batch = train_batch.to(device)

            train_loss = torch.mean(model.loss_function(train_batch))
            iteration_losses[n_iterations_done] = train_loss.item()

            optimizer.zero_grad()
            train_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=200.) # Gradient clipping

            optimizer.step()

            # We validate first iteration, every n iterations, and at the last iteration
            if (n_iterations_done % validate_every_n_iterations) == 0 or (n_iterations_done == n_iterations - 1):
                model.eval()

                with torch.no_grad():
                    val_loss_averager = make_averager()

                    for validation_batch, _ in validation_dataloader:
                        validation_batch = dequantize(validation_batch)
                        validation_batch = validation_batch.to(device)

                        val_loss = torch.mean(model.loss_function(validation_batch))
                        val_loss_averager(val_loss.item())

                    validation_losses.append(val_loss_averager(None))

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


                    if n_iterations_without_improvements >= early_stopping_threshold:
                        stop = True
                        break


            n_iterations_done += 1
            model.train()
            if n_iterations_done >= n_iterations:
                stop = True
                break

# Plot train and test loss
plt.figure()
plt.plot(range(n_iterations), iteration_losses)
plt.plot(range(0, n_iterations, validate_every_n_iterations), validation_losses)

print('Evaluating...')

# We calculate final results on the best model
model.load_state_dict(torch.load(f'checkpoints/{run_name}_best.pt'))
model.eval()

test_loss_averager = make_averager()
with torch.no_grad():

    # Plot samples from best model
    samples = model.sample(16)
    samples = samples.cpu().detach()
    fig = plot_image_grid(samples, cols=4)
    plt.show()

    # Calculate test loss
    for test_batch, _ in test_dataloader:
        test_batch = dequantize(test_batch)
        test_batch = test_batch.to(device)
        if model_name == 'maf':
            test_batch = test_batch.view(-1, torch.prod(torch.tensor(image_dim)))
        test_batch_loss = torch.mean(model.loss_function(test_batch))
        test_loss_averager(test_batch_loss.item())
    test_loss = test_loss_averager(None)

    print(f'Test loss: {test_loss}')

    # This might run out of memory on smaller GPUs

    # Approximate log likelihood if model not exact
    try:
        if has_importance_sampling(model):
            print(f'Approximating log-likelihood of test set...')
            if model_name == 'aef-linear':
                sigma_importance = get_posterior_scale_aef_linear(dataset, latent_dims)
                test_ll_averager = make_averager()
                for test_batch, _ in test_dataloader:
                    test_batch = dequantize(test_batch)
                    test_batch = test_batch.to(device)
                    for iw_iter in range(20):
                        log_likelihood = torch.mean(model.approximate_marginal(test_batch, std=sigma_importance,
                                                                               n_samples=128))
                        test_ll_averager(log_likelihood.item())
                test_ll = test_ll_averager(None)
            else:
                test_ll_averager = make_averager()
                for test_batch, _ in test_dataloader:
                    test_batch = dequantize(test_batch)
                    test_batch = test_batch.to(device)
                    for iw_iter in range(20):
                        log_likelihood = torch.mean(model.approximate_marginal(test_batch, n_samples=128))
                        test_ll_averager(log_likelihood.item())
                test_ll = test_ll_averager(None)
            # We only add this value to the summary if we approximate the log likelihood (since we provide test_loss
            # in both cases).
        else:
            bpp_test = bits_per_pixel(test_loss, n_pixels)
            bpp_test_adjusted = bits_per_pixel(test_loss, n_pixels, adjust_value=256.)
        print(f'BPP (test set): {bpp_test_adjusted}')
    except Exception as e:
        print(f'Failed to approximate likelihood due to error below.')
        print(e)
        traceback.print_exc()

    try:
        # Calculate FID
        print(f'Calculating FID score...')
        incept = InceptionV3().to(device)
        fid = calculate_fid(model, dataset, device, batch_size=128, incept=incept, data_dir=args.data_dir)
        print(f'FID: {fid}')
    except Exception as e:
        print(f'Failed to calculate FID due to error below.')
        print(e)
        traceback.print_exc()



    print(test_loss, test_ll)


