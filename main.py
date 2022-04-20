import matplotlib.pyplot as plt

from util import get_avg_loss_over_iterations, plot_loss_over_iterations
import normflow as nf
import torch

import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from tqdm import tqdm
from flows.realnvp import RealNVP
from models.autoencoder import Encoder, Decoder
from models.normalizing_autoencoder import NormalizingAutoEncoder
from util import make_averager, refresh_bar, plot_loss, dequantize
from datasets import get_train_val_dataloaders

import wandb

def main():

    run = wandb.init(project="test-project", entity="nae", name=None) #todo: name should be defined with command line arguments
                                                                            #todo: example {model}_{dataset}_{latent_space_dim}_{run_number}
    # 2-d latent space, parameter count in same order of magnitude
    # as in the original VAE paper (VAE paper has about 3x as many)
    model_name = 'test'
    n_iterations = 200
    dataset = 'mnist'
    latent_dims = 4
    batch_size = 128
    capacity = 64
    learning_rate = 1e-3
    variational_beta = 1
    use_gpu = True
    validate_every_n_iterations = 50

    wandb.config = {
        "learning_rate": learning_rate,
        "n_iterations": n_iterations,
        "batch_size": batch_size,
        "dataset": dataset,
        "latent_dims": latent_dims
    }
    if dataset == 'mnist':
        alpha = 1e-6

    device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")

    do_dequantize = True

    p_validation = 0.1
    train_dataloader, validation_dataloader, image_dim, alpha = get_train_val_dataloaders('mnist', batch_size, p_validation)

    core_flow = RealNVP(input_dim=latent_dims, num_flows=6, hidden_units=256)
    encoder = Encoder(64,latent_dims, image_dim)
    decoder = Decoder(64, latent_dims, image_dim)
    mask = torch.zeros(28,28)
    mask[13:15, 13:15] = 1
    mask = mask.to(device)
    preprocessing_layers = [nf.transforms.Logit(alpha), nf.flows.ActNorm(image_dim)]
    model = NormalizingAutoEncoder(core_flow, encoder, decoder, mask, preprocessing_layers)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)#, weight_decay=1e-5)

    model = model.to(device)

    print('Training ...')

    stop = False
    n_iterations_done = 0
    n_times_validated = 0
    iteration_losses = np.zeros((n_iterations, ))
    validation_losses = []
    validation_iterations = []
    averaging_window_size = 10

    with tqdm(total=n_iterations, desc="iteration [loss: ...]") as iterations_bar:
        while not stop:
            for image_batch, _ in train_dataloader:
                if do_dequantize:
                    image_batch = dequantize(image_batch)
                image_batch = image_batch.to(device)

                loss = torch.mean(model.neg_log_likelihood(image_batch))
                iteration_losses[n_iterations_done] = loss.item()
                metrics = {
                    'train/train_loss': loss
                }

                # backpropagation
                optimizer.zero_grad()
                loss.backward()

                # one step of the optmizer
                optimizer.step()

                wandb.watch(model)

                refresh_bar(iterations_bar,
                            f"iteration [loss: "
                            f"{get_avg_loss_over_iterations(iteration_losses, averaging_window_size, n_iterations_done):.3f}]")

                # We validate first epoch
                if (n_iterations_done % validate_every_n_iterations) == 0 or (n_iterations_done == n_iterations - 1):
                    val_batch_bar = tqdm(validation_dataloader, leave=False, desc='validation batch',
                                         total=len(validation_dataloader))
                    val_loss_averager = make_averager()
                    with torch.no_grad():
                        for validation_batch, _ in val_batch_bar:
                            if do_dequantize:
                                validation_batch = dequantize(validation_batch)
                            validation_batch = validation_batch.to(device)
                            loss = torch.mean(model.neg_log_likelihood(validation_batch))
                            refresh_bar(val_batch_bar,
                                        f"validation batch [loss: {val_loss_averager(loss.item()):.3f}]")
                        validation_losses.append(val_loss_averager(None))
                        val_metrics = {
                            'val/val_loss': validation_losses[-1]
                        }
                        wandb.log({**metrics, **val_metrics})
                        if n_iterations_done == 0:
                            best_loss = validation_losses[-1]
                            best_it = n_iterations_done
                        elif validation_losses[-1] < best_loss:
                            best_loss = validation_losses[-1]
                            torch.save(model.state_dict(), f'checkpoints/{model_name}_best.pt')
                            best_it = n_iterations_done
                        validation_iterations.append(n_iterations_done)
                        torch.save({
                            'n_iterations_done': n_iterations_done,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'iteration_losses': iteration_losses,
                            'validation_losses': validation_losses
                                    },
                        f'checkpoints/{model_name}_latest.pt')
                        n_times_validated += 1

                else:
                    wandb.log(metrics)

                n_iterations_done += 1
                iterations_bar.update(1)
                if n_iterations_done >= n_iterations:
                    stop = True
                    break

    plot_loss_over_iterations(iteration_losses, validation_losses,
                              validation_iterations)
    wandb.summary['best_iteration'] = best_it
    wandb.summary['best_val_loss'] = best_loss
    artifact_best = wandb.Artifact('model_best', type='model')
    artifact_best.add_file(f'checkpoints/{model_name}_best.pt')
    run.log_artifact(artifact_best)
    artifact_latest = wandb.Artifact('model_latest', type='model')
    artifact_latest.add_file(f'checkpoints/{model_name}_latest.pt')
    run.log_artifact(artifact_latest)
    run.finish()

    for i in range(4):
        samples = model.sample(16)
        samples = samples.cpu().detach().numpy()
        _, axs = plt.subplots(4, 4, )
        axs = axs.flatten()
        for img, ax in zip(samples, axs):
            ax.axis('off')
            ax.imshow(img.reshape(28, 28), cmap='gray')

        plt.show()

if __name__ == "__main__":
    main()
