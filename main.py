import matplotlib.pyplot as plt
import numpy as np
import torch
from nflows.transforms.base import InverseTransform
from nflows.transforms.nonlinearities import Sigmoid
from nflows.transforms.standard import AffineTransform

import wandb
from datasets import get_train_val_dataloaders, get_test_dataloader
from flows.actnorm import ActNorm
from models.autoencoder import ConvolutionalEncoder, IndependentVarianceDecoder
from models.normalizing_autoencoder import NormalizingAutoEncoder
from util import make_averager, dequantize


def main():
    # 2-d latent space, parameter count in same order of magnitude
    # as in the original VAE paper (VAE paper has about 3x as many)
    model_name = 'test'
    n_iterations = 10000
    dataset = 'mnist'
    latent_dims = 4
    batch_size = 128
    learning_rate = 1e-4
    use_gpu = True
    validate_every_n_iterations = 500

    config = {
        "learning_rate": learning_rate,
        "n_iterations": n_iterations,
        "batch_size": batch_size,
        "dataset": dataset,
        "latent_dims": latent_dims
    }
    run = wandb.init(project="test-project", entity="nae",
                     name=None, config=config)  # todo: name should be defined with command line arguments
    # todo: example {model}_{dataset}_{latent_space_dim}_{run_number}

    #device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    p_validation = 0.1
    train_dataloader, validation_dataloader, image_dim, alpha = get_train_val_dataloaders('mnist', batch_size,
                                                                                          p_validation)
    test_dataloader = get_test_dataloader('mnist', batch_size)

    preprocessing_layers = [InverseTransform(AffineTransform(alpha, 1 - 2 * alpha)), Sigmoid(), ActNorm(1)]
    encoder = ConvolutionalEncoder(hidden_channels=64, input_shape=image_dim, latent_dim=latent_dims)
    decoder = IndependentVarianceDecoder(hidden_channels=64, output_shape=image_dim, latent_dim=latent_dims)
    model = NormalizingAutoEncoder(encoder, decoder,
                                   preprocessing_layers=preprocessing_layers)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

    model = model.to(device)

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
                            loss = torch.mean(model.neg_log_likelihood(validation_batch))
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
                            torch.save(model.state_dict(), f'checkpoints/{model_name}_best.pt')
                            best_it = n_iterations_done
                        validation_iterations.append(n_iterations_done)
                        torch.save({
                            'n_iterations_done': n_iterations_done,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'iteration_losses': iteration_losses,
                            'validation_losses': validation_losses,
                            'best_loss': best_loss},
                            f'checkpoints/{model_name}_latest.pt')
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

    model.load_state_dict(torch.load(f'checkpoints/{model_name}_best.pt'))
    model.eval()
    test_loss_averager = make_averager()
    with torch.no_grad():
        for test_batch, _ in test_dataloader:
            test_batch = dequantize(test_batch)
            test_batch = test_batch.to(device)
            loss = torch.mean(model.neg_log_likelihood(test_batch))
            test_loss_averager(loss.item())
        test_loss = test_loss_averager(None)

        for i in range(4):
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
    artifact_best.add_file(f'checkpoints/{model_name}_best.pt')
    run.log_artifact(artifact_best)
    artifact_latest = wandb.Artifact('model_latest', type='model')
    artifact_latest.add_file(f'checkpoints/{model_name}_latest.pt')
    run.log_artifact(artifact_latest)
    wandb.summary['best_iteration'] = best_it
    wandb.summary['test_loss'] = test_loss

    run.finish()


if __name__ == "__main__":
    main()
