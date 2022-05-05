from typing import Mapping, Union, Optional, Callable, List
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.distributions.normal import Normal
import wandb

import os
import torchvision

from models.model_database import get_model


def get_avg_loss_over_iterations(iteration_losses: np.array, window_size: int, cur_iteration: int):
    low_window = max(0, cur_iteration - window_size)
    high_window = cur_iteration + 1
    return np.mean(iteration_losses[low_window:high_window])

def plot_loss_over_iterations(iterations_losses: np.array, val_losses: List = None,
                              val_iterations: List = None, window_size: int = 10):
    def moving_average(a, n=window_size):
        ret = np.cumsum(a, dtype=float)
        ret[n:] = (ret[n:] - ret[:-n])/n
        ret[:n] = ret[:n] / np.arange(1, n+1)
        return ret


    plt.figure()

    plt.plot(np.arange(len(iterations_losses)), moving_average(iterations_losses), color='tab:blue')
    if val_losses is not None:
        plt.plot(val_iterations, val_losses, color='tab:orange')
    plt.legend(["Training loss"] if val_losses is None else ["Training loss", "Validation loss"])
    plt.xlabel("Iteration")
    plt.ylabel("Negative log likelihood")
    plt.show()

def make_averager() -> Callable[[Optional[float]], float]:
    """ Returns a function that maintains a running average
    :returns: running average function
    """
    count = 0
    total = 0

    def averager(new_value: Optional[float]) -> float:
        """ Running averager
        :param new_value: number to add to the running average,
                          if None returns the current average
        :returns: the current average
        """
        nonlocal count, total
        if new_value is None:
            return total / count if count else float("nan")
        count += 1
        total += new_value
        return total / count

    return averager


def save_in_dataframe(df_log, labels, mus, stddevs, epoch):
    df = pd.DataFrame()

    df['index'] = np.arange(len(mus[:, 0])) * epoch
    df['image_ind'] = np.arange(len(mus[:, 0]))
    df['class'] = labels.data.numpy().astype(str)
    df['mu_x'] = mus[:, 0]
    df['mu_y'] = mus[:, 1]
    df['std_x'] = stddevs[:, 0]
    df['std_y'] = stddevs[:, 1]
    df['epoch'] = np.ones(len(mus[:, 0])) * epoch

    df_log = pd.concat([df_log, df])

    return df_log


def run_on_testbatch(df_log, vae, epoch, x, y, device=None):
    with torch.no_grad():
        if device is not None:
            x = x.to(device)
        x, mus, stddevs = vae(x)
        x = x.to('cpu')
        mus = mus.to('cpu').data.numpy()
        stddevs = stddevs.to('cpu').mul(0.5).exp_().data.numpy()

    return save_in_dataframe(df_log, y, mus, stddevs, epoch)

def plot_loss(train_loss, val_loss=None):
    plt.figure()
    plt.plot(np.arange(len(train_loss)), train_loss, color='tab:blue')
    if val_loss is not None:
        plt.plot(np.arange(len(val_loss)), val_loss, color='tab:orange')
    plt.legend(["Training loss"] if val_loss is None else ["Training loss", "Validation loss"])
    plt.xlabel("Epoch")
    plt.ylabel("Negative log likelihood")
    plt.show()

def refresh_bar(bar, desc):
    bar.set_description(desc)
    bar.refresh()

def dequantize(batch): # TODO: move somewhere else
    noise = torch.rand(*batch.shape)
    batch = (batch * 255. + noise) / 256.
    return batch

def vae_log_prob(vae, images, n_samples):
    '''
    Implementation of importance sampling marginal likelihood for VAEs
    :return:
    '''
    #todo: needs further testing
    batch_size = images.shape[0]
    mu_z, sigma_z = vae.encode(images)
    samples = Normal(mu_z, sigma_z).sample([n_samples]).transpose(1,0)
    mu_x, sigma_x = vae.decode(samples.reshape(batch_size*n_samples, -1))
    mu_x, sigma_x = mu_x.view(batch_size, n_samples, -1), sigma_x.view(batch_size, n_samples, -1)
    p_x_z = Normal(mu_x, sigma_x).log_prob(images.view(batch_size,1, -1)).sum([2]).view(batch_size,n_samples)
    p_latent = Normal(0,1).log_prob(samples).sum([-1])
    q_latent = Normal(mu_z.unsqueeze(1), sigma_z.unsqueeze(1)).log_prob(samples).sum([-1])

    #return torch.log(torch.mean(torch.exp(p_x_z+p_latent-q_latent)))
    return torch.mean(torch.logsumexp(p_x_z + p_latent - q_latent, [1]) - torch.log(torch.tensor(n_samples)))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def download_wandb_artifact(run, project_name, experiment_name, download_best=True, version='latest'):
    if download_best:
        artifact = run.use_artifact(f'nae/{project_name}/{experiment_name}_best:{version}', type='model')
    else:
        artifact = run.use_artifact(f'nae/{project_name}/{experiment_name}_latest:{version}', type='model')
    artifact_dir = artifact.download()
    return artifact_dir


def download_artifact_and_get_path(run, project_name, experiment_name, download_best=True, version='latest'):
    artifact_dir = download_wandb_artifact(run, project_name, experiment_name, download_best, version)
    return artifact_dir + '/' + os.listdir(artifact_dir)[0]



def load_best_model(run, project_name, model_name, experiment_name, device, latent_dims, image_dim, alpha,
                    decoder, architecture_size, prior_flow, posterior_flow, version='latest'):
    model = get_model(model_name, architecture_size, decoder, latent_dims, image_dim, alpha, posterior_flow, prior_flow) # needed as some components such as actnorm need to be initialized
    model.loss_function(model.sample(10)) # needed as some components such as actnorm need to be initialized

    model_path = download_artifact_and_get_path(run, project_name, experiment_name, download_best=True, version=version)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    return model

def load_latest_model(run, project_name, experiment_name, device, model, optimizer, validate_every_n_iterations, version='latest'):
    model_path = download_artifact_and_get_path(run, project_name, experiment_name, download_best=False, version=version)
    checkpoint = torch.load(model_path, map_location=torch.device(device))
    n_iterations_done = checkpoint['n_iterations_done']
    iteration_losses = checkpoint['iteration_losses']
    validation_losses = checkpoint['validation_losses']
    best_loss = checkpoint['best_loss']
    model.loss_function(model.sample(2))
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    j = 0
    for i in range(n_iterations_done):
        log_dict = {'train_loss': iteration_losses[i]}
        if i % validate_every_n_iterations == 0:
            log_dict['val_loss'] = validation_losses[j]
            j+=1
        run.log(log_dict)
    return n_iterations_done, iteration_losses, validation_losses, best_loss, model, optimizer

def plot_image_grid(samples, cols, padding=2, title=None, hires=False):
    '''
    Samples should be a torch aray with dimensions BxCxWxH
    '''
    if hires:
        fig = plt.figure(figsize=(10, 10), dpi=300)
    else:
        fig = plt.figure()
    grid_img = torchvision.utils.make_grid(samples, padding=padding, pad_value=1., nrow=cols)
    plt.imshow(grid_img.permute(1, 2, 0))
    plt.axis("off")
    if title:
        plt.suptitle(title)
    return fig

        
def bits_per_pixel(neg_log_prob, n_pixels, adjust_value=None):
    if adjust_value:
        neg_log_prob += (n_pixels*torch.log(torch.ones(1)*adjust_value))[0]
    log_prob_base_2 = neg_log_prob/torch.log(torch.ones(1)*2.)

    return log_prob_base_2/n_pixels

