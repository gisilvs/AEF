import os
import traceback

import numpy as np
import torch
import torch.nn.functional as F

import metrics
import util
import wandb
import pandas as pd

from models.model_database import get_model
from util import load_best_model, vae_log_prob, make_averager, dequantize, bits_per_pixel, has_importance_sampling
from datasets import get_test_dataloader, get_train_val_dataloaders
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)

def extract_data_from_runs(project_name, runs_finished=True):
    api = wandb.Api(timeout=19)

    model_names = []
    run_nr = []
    dataset = []
    architecture_size = []  # May not exist for phase 1
    prior_flow = []  # May not exist for phase 1
    posterior_flow = []  # May not exist for phase 1
    decoder = []
    latent_dims = []
    test_loss = []
    test_bpp = []
    test_bpp_adjusted = []
    train_loss = []
    val_loss = []
    test_rce_with_noise = []
    test_rce = []
    fid = []
    noise_level = []
    preprocessing = []
    ife = []


    runs = api.runs(path=f"nae/{project_name}")
    for run in runs:
        if run.state == 'running':
            continue
        if runs_finished and run.state != 'finished':
            continue
        model_names.append(get_field_from_config(run, "model"))
        dataset.append(get_field_from_config(run, "dataset"))
        decoder.append(get_field_from_config(run, "decoder"))
        latent_dims.append(get_field_from_config(run, "latent_dims", type="int"))
        architecture_size.append(get_field_from_config(run, "architecture_size"))
        prior_flow.append(get_field_from_config(run, "prior_flow"))
        posterior_flow.append(get_field_from_config(run, "posterior_flow"))
        test_loss.append(get_field_from_summary(run, "test_loss", type="float"))
        test_bpp.append(get_field_from_summary(run, "test_bpp", type="float"))
        test_bpp_adjusted.append(get_field_from_summary(run, "test_bpp_adjusted", type="float"))
        train_loss.append(get_field_from_summary(run, "train_loss", type="float"))
        val_loss.append(get_field_from_summary(run, "val_loss", type="float"))
        test_rce_with_noise.append(get_field_from_summary(run, "test_rce_with_noise", type="float"))
        test_rce.append(get_field_from_summary(run, "test_rce", type="float"))
        fid.append(get_field_from_summary(run, "fid", type="float"))
        noise_level.append(get_field_from_config(run, "noise_level", type="float"))
        preprocessing.append(get_field_from_config(run, "preprocessing"))
        ife.append(get_field_from_summary(run, "ife", type="float"))

    col_dict = {'model': model_names,
                'dataset': dataset,
                'latent_dims': latent_dims,
                'decoder': decoder,
                'test_loss': test_loss,
                'test_bpp': test_bpp,
                'test_bpp_adjusted': test_bpp_adjusted,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'architecture_size': architecture_size,
                'prior_flow': prior_flow,
                'posterior_flow': posterior_flow,
                'test_rce_with_noise': test_rce_with_noise,
                'test_rce': test_rce,
                'fid': fid,
                'noise_level': noise_level,
                'preprocessing': preprocessing,
                'ife': ife
                }
    df = pd.DataFrame(col_dict)

    return df


def get_field_from_config(run: wandb.run, field: str, type: str = 'str'):
    if field not in run.config.keys():
        return None
    if type == 'int':
        return int(run.config[field])
    elif type == 'float':
        return float(run.config[field])
    else:
        return run.config[field]

def get_field_from_summary(run: wandb.run, field: str, type: str = 'str'):
    if field not in run.summary.keys() or run.summary[field] is None:
        return None
    if type == 'int':
        return int(run.summary[field])
    elif type == 'float':
        return float(run.summary[field])
    else:
        return run.summary[field]


def add_ife():
    use_gpu = True
    device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")
    image_dim = [3, 32, 32]

    alpha = 0.05

    batch_size = 128
    api = wandb.Api()
    project_name = 'denoising-experiments-6'
    runs = api.runs(path=f"nae/{project_name}")
    data_dir = 'celebahq'
    incept = metrics.InceptionV3().to(device)
    for run in runs:

        try:
            run_name = run.name
            model_name = get_field_from_config(run, "model")
            dataset = get_field_from_config(run, "dataset")

            decoder = get_field_from_config(run, "decoder")
            latent_dims = get_field_from_config(run, "latent_dims", type="int")

            architecture_size = get_field_from_config(run, "architecture_size")

            posterior_flow = get_field_from_config(run, "posterior_flow")
            if posterior_flow is None:
                posterior_flow = 'none'
            prior_flow = get_field_from_config(run, "prior_flow")
            if prior_flow is None:
                prior_flow = 'none'

            noise_level = get_field_from_config(run, "noise_level")
            torch.manual_seed(3)  # Seed noise for equal test comparison
            noise_distribution = torch.distributions.normal.Normal(torch.zeros(batch_size, *image_dim).to(device),
                                                                   noise_level * torch.ones(batch_size, *image_dim).to(
                                                                       device))

            model = get_model(model_name, architecture_size, decoder, latent_dims, image_dim, alpha, posterior_flow,
                              prior_flow)

            #model.loss_function(model.sample(10))  # needed as some components such as actnorm need to be initialized
            artifact = api.artifact(f'nae/{project_name}/{run_name}_best:latest')
            artifact_dir = artifact.download()
            artifact_dir = artifact_dir + '/' + os.listdir(artifact_dir)[0]
            model.load_state_dict(torch.load(artifact_dir, map_location=device))
            model = model.to(device)

            model = model.eval()

            try:
                ife = metrics.calculate_ife(model, dataset, device, noise_distribution, batch_size=batch_size, incept=incept, data_dir=data_dir)
                run.summary['ife'] = ife
                run.summary.update()
                print(f'{run_name} updated.')

            except Exception as e:
                print(e)
                traceback.print_exc()
                print(f'Failed FID in {run_name}')

        except Exception as e:
            print(e)
            traceback.print_exc()
            print(f'Failed to update {run_name}')
            continue


def generate_phase1_table(df, latent_dims=2):

    datasets = ['mnist', 'fashionmnist', 'kmnist']
    models = ['vae', 'aef-linear', 'aef-center', 'aef-corner']


    for dataset_idx, dataset in enumerate(datasets):
        print(dataset)
        for model_idx, model_name in enumerate(models):

            runs = df.loc[(df.loc[:, 'model'] == model_name) & (df.loc[:, 'dataset'] == dataset) & (df.loc[:, 'latent_dims'] == latent_dims)]
            mean_bpp = runs.loc[:, 'test_bpp_adjusted'].mean(axis=0)
            se_bpps = runs.loc[:, 'test_bpp_adjusted'].sem(axis=0)
            mean_fid = runs.loc[:, 'fid'].mean(axis=0)
            se_fid = runs.loc[:, 'fid'].sem(axis=0)
            print(f'{model_name} BPP {mean_bpp} +- {se_bpps}')
            print(f'{model_name} FID {mean_fid} +- {se_fid}')

def generate_phase2_table(df, latent_dims=2):

    datasets = ['celebahq']
    models = ['vae', 'aef-linear']
    print(f'latents {latent_dims}')
    for dataset_idx, dataset in enumerate(datasets):
        print(dataset)
        for model_idx, model_name in enumerate(models):
            runs = df.loc[(df.loc[:, 'model'] == model_name) & (df.loc[:, 'dataset'] == dataset) & (df.loc[:, 'latent_dims'] == latent_dims)]
            mean_bpp = runs.loc[:, 'test_bpp_adjusted'].mean(axis=0)
            se_bpps = runs.loc[:, 'test_bpp_adjusted'].sem(axis=0)
            mean_fid = runs.loc[:, 'fid'].mean(axis=0)
            se_fid = runs.loc[:, 'fid'].sem(axis=0)
            print(f'{model_name} BPP {mean_bpp} +- {se_bpps}')
            print(f'{model_name} FID {mean_fid} +- {se_fid}')

def generate_denoising_table(df):

    datasets = ['mnist', 'fashionmnist', 'kmnist']
    models = ['ae', 'vae', 'aef-linear']

    noise_levels = [0.25, 0.5, 0.75, 1.0]
    latent_sizes = [2, 32]
    for latent_dims in latent_sizes:
        print(latent_dims)
        for dataset_idx, dataset in enumerate(datasets):
            for model_idx, model_name in enumerate(models):
                    for noise_idx, noise_level in enumerate(noise_levels):
                        runs = df.loc[(df.loc[:, 'latent_dims'] == latent_dims) & (df.loc[:, 'model'] == model_name) & (df.loc[:, 'dataset'] == dataset) & (df.loc[:, 'noise_level'] == noise_level)]
                        val = runs.loc[:, 'ife'].mean(axis=0)
                        print(f'{dataset} {model_name} sigma {noise_level} IFE: {val}')


def generate_denoising_table_celeba(df):

    models = ['ae', 'vae', 'aef-linear']
    noise_levels = [0.05, 0.1, 0.2]

    for model_idx, model_name in enumerate(models):
        for noise_idx, noise_level in enumerate(noise_levels):

            runs = df.loc[(df.loc[:, 'model'] == model_name) & (df.loc[:, 'noise_level'] == noise_level)]
            val = runs.loc[:, 'ife'].mean(axis=0)
            print(f'{model_name} sigma {noise_level}: {val}')


def find_sigma():

    runs_per_latent = 100

    model_name = 'aef-linear'

    latent_sizes = [128]
    project_name = 'ablation-celeba-big'
    dataset = 'celebahq'
    data_dir = 'celebahq'
    #data_dir = 'data/celebahq64'
    img_dim = [3, 32, 32]
    alpha = 0.05
    n_batches = 16 * 4
    iw_batch_size = 32
    val_batch_size = 16
    val_batches = 128

    api = wandb.Api()
    use_gpu = True
    device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")

    sigmas = [0.1, 0.01, 0.001, 0.0001]

    print(dataset)
    print(f'n_batches: {n_batches}')
    print(f'batch_size: {iw_batch_size}')
    for latent_idx, latent_dims in enumerate(latent_sizes):
        print(f"Latent dims: {latent_dims}")
        runs = api.runs(path=f"nae/{project_name}", filters={
            "config.latent_dims": latent_dims,
            "config.model": model_name,
            "config.dataset": dataset,
        })
        runs_done_latent = 0
        for run_idx, run in enumerate(runs):
            experiment_name = run.name

            print(experiment_name)

            decoder = get_field_from_config(run, "decoder")
            latent_dims = get_field_from_config(run, "latent_dims", type="int")

            posterior_flow = get_field_from_config(run, 'posterior_flow')
            if posterior_flow is None:
                posterior_flow = 'none'

            prior_flow = get_field_from_config(run, 'prior_flow')
            if prior_flow is None:
                prior_flow = 'none'
            architecture_size = get_field_from_config(run, "architecture_size")
            model = get_model(model_name, architecture_size, decoder, latent_dims, img_dim, alpha,
                                     posterior_flow,
                                     prior_flow)
            run_name = run.name
            artifact = api.artifact(
                f'nae/{project_name}/{run_name}_best:latest')  # run.restore(f'{run_name}_best:latest', run_path=run.path, root='./artifacts')
            artifact_dir = artifact.download()
            artifact_dir = artifact_dir + '/' + os.listdir(artifact_dir)[0]
            model.load_state_dict(torch.load(artifact_dir, map_location=device))
            model = model.to(device)

            model.eval()


            with torch.no_grad():
                print('Calculating val loss')
                _, validation_dataloader, image_dim, _ = get_train_val_dataloaders(dataset, val_batch_size,
                                                                                   data_dir=data_dir)
                val_loss_averager = make_averager()
                val_batches_done = 0
                for batch, _ in validation_dataloader:
                    batch = dequantize(batch)
                    batch = batch.to(device)
                    batch_loss = torch.mean(model.loss_function(batch))
                    val_loss_averager(batch_loss.item())
                    val_batches_done += 1
                    if val_batches_done >= val_batches:
                        break
                print(f'Validation loss: {val_loss_averager(None)}.')

                _, validation_dataloader, image_dim, _ = get_train_val_dataloaders(dataset, iw_batch_size,
                                                                                   data_dir=data_dir)
                for sigma in sigmas:
                    print(f'Sigma {sigma}: ', end='')
                    approximate_ll_averager = make_averager()

                    n_batches_done = 0
                    count_nans_iw = 0

                    for batch, _ in validation_dataloader:
                        batch = dequantize(batch)
                        batch = batch.to(device)

                        for iw_iter in range(10):
                            try:
                                log_likelihood = torch.mean(model.approximate_marginal(batch, n_samples=64, std=sigma))
                                approximate_ll_averager(log_likelihood.item())
                            except Exception as E:
                                count_nans_iw += 1

                                # print(E)
                                # traceback.print_exc()
                                # return
                                continue
                            if count_nans_iw > 40:
                                break

                        n_batches_done += 1
                        if n_batches_done >= n_batches:
                            break
                    if count_nans_iw > 40:
                        print('too many NANs.')
                        continue
                    approximate_ll = approximate_ll_averager(None)
                    print(f'Sigma {sigma}: {approximate_ll} ll. NANs: {count_nans_iw}')

            runs_done_latent += 1
            if runs_done_latent >= runs_per_latent:
                break


def add_ll_phase1():
    use_gpu = True
    device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")
    project_name = 'phase1'
    image_dim = [1, 28, 28]
    n_pixels = np.prod(image_dim)

    alpha = 1e-6

    batch_size = 128

    api = wandb.Api()
    for dataset in ['mnist', 'fashionmnist', 'kmnist']:
        test_dataloader = get_test_dataloader(dataset, batch_size)
        runs = api.runs(path=f"nae/{project_name}", filters={
            "config.model": 'aef-linear',
            "config.dataset": dataset,
        })
        for run in runs:
            try:
                model_name = get_field_from_config(run, "model")

                dataset = get_field_from_config(run, "dataset")

                decoder = get_field_from_config(run, "decoder")
                latent_dims = get_field_from_config(run, "latent_dims", type="int")

                architecture_size = get_field_from_config(run, "architecture_size")
                if architecture_size is None:
                    architecture_size = "small"

                posterior_flow = get_field_from_config(run, "posterior_flow")
                if posterior_flow is None:
                    posterior_flow = 'none'
                prior_flow = get_field_from_config(run, "prior_flow")
                if prior_flow is None:
                    prior_flow = 'none'

                model = get_model(model_name, architecture_size, decoder, latent_dims, image_dim, alpha, posterior_flow,
                                  prior_flow)
                #model.loss_function(model.sample(10))  # needed as some components such as actnorm need to be initialized
                run_name = run.name
                artifact = api.artifact(f'nae/{project_name}/{run_name}_best:latest')
                artifact_dir = artifact.download()
                artifact_dir = artifact_dir + '/' + os.listdir(artifact_dir)[0]
                model.load_state_dict(torch.load(artifact_dir, map_location=device))
                model = model.to(device)

                model = model.eval()

                importance_std = util.get_posterior_scale_aef_linear(dataset, latent_dims)

                with torch.no_grad():
                    # Approximate log likelihood if model in VAE family

                    if has_importance_sampling(model):

                        test_ll_averager = make_averager()
                        for test_batch, _ in test_dataloader:
                            test_batch = dequantize(test_batch)
                            test_batch = test_batch.to(device)
                            for iw_iter in range(20):
                                log_likelihood = torch.mean(model.approximate_marginal(test_batch, n_samples=128, std=importance_std))
                                test_ll_averager(log_likelihood.item())
                        test_ll = test_ll_averager(None)
                        # We only add this value to the summary if we approximate the log likelihood (since we provide test_loss
                        # in both cases).

                        bpp_test = bits_per_pixel(test_ll, n_pixels)
                        bpp_test_adjusted = bits_per_pixel(test_ll, n_pixels, adjust_value=256.)

                        run.summary['test_log_likelihood'] = test_ll
                        run.summary['test_bpp'] = bpp_test
                        run.summary['test_bpp_adjusted'] = bpp_test_adjusted
                        run.summary.update()
                        print(f"Updated {run_name}")
                    else:
                        print('Something went wrong')

            except Exception as e:
                print(e)
                traceback.print_exc()
                print(f'Failed to update {run_name}')
                continue


if __name__ == '__main__':
    sys.exit(0)