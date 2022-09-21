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

def extract_data_from_runs(project_name='phase1', runs_finished=True):
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

        model_name = get_field_from_config(run, "model")
        model_names.append(model_name)

        run_nr_idx = run.name.find('run_')
        if run_nr_idx != -1:
            run_nr.append(int(run.name[run_nr_idx + 4]))
        else:
            run_nr.append(None)

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


def update_nae_external():
    model_name = 'nae-external'
    api = wandb.Api(timeout=59)
    alpha = 1e-6
    img_dim = [1, 28, 28]
    n_pixels = np.prod(img_dim)
    project_name = 'phase1'

    use_gpu = True
    device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")


    runs = api.runs(path=f"nae/{project_name}", filters={#"config.dataset": dataset,
                                                             #"config.latent_dims": latent_dims,
                                                             "config.model": model_name,
                                                             # "config.posterior_flow": posterior_flow,
                                                             # "config.prior_flow": prior_flow,
                                                             #"config.noise_level": noise_level
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

            model = get_model(model_name, architecture_size, decoder, latent_dims, img_dim, alpha, posterior_flow,
                              prior_flow)

            run_name = run.name
            artifact = api.artifact(f'nae/{project_name}/{run_name}_best:latest')
            artifact_dir = artifact.download()
            artifact_dir = artifact_dir + '/' + os.listdir(artifact_dir)[0]
            model.load_state_dict(torch.load(artifact_dir, map_location=device))
            model = model.to(device)

            test_loader = get_test_dataloader(dataset, batch_size=128)
            model = model.eval()

            test_ll_averager = make_averager()
            for test_batch, _ in test_loader:
                test_batch = dequantize(test_batch)
                test_batch = test_batch.to(device)
                for iw_iter in range(20):
                    log_likelihood = torch.mean(model.negative_log_likelihood(test_batch, n_samples=128))
                    test_ll_averager(log_likelihood.item())
            new_test_ll = test_ll_averager(None)
            new_bpp_test = bits_per_pixel(new_test_ll, n_pixels)
            new_bpp_test_adjusted = bits_per_pixel(new_test_ll, n_pixels, adjust_value=256.)

            run.summary["test_log_likelihood"] = new_test_ll
            run.summary["test_bpp"] = new_bpp_test
            run.summary['test_bpp_adjusted'] = new_bpp_test_adjusted
            run.summary.update()
            print(f"Updated {run_name}")
        except Exception as E:
            print(E)
            print(f'Failed to update bpp/likelihood of {run.name}')
            traceback.print_exc()
            continue

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



def update_log_likelihood():
    use_gpu = True
    device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")
    project_name = 'phase1'
    image_dim = [1, 28, 28]
    n_pixels = np.prod(image_dim)
    vae_like_models = ['vae-iaf', 'vae', 'iwae']

    alpha = 1e-6

    api = wandb.Api()
    runs = api.runs(path="nae/phase1")
    architecture_size = "small"
    for run in runs:
        try:
            model_name = get_field_from_config(run, "model")
            # Backwards compatibility: before we used 'nae' for both 'nae-center' and 'nae-corner'.
            if model_name == 'nae':
                if run.name.split('_')[-1] == 'corner':
                    model_name = 'nae-corner'
                    run.config["model"] = model_name
                    run.update()
                elif run.name.split('_')[-1] == 'center':
                    model_name = 'nae-center'
                    run.config["model"] = model_name
                    run.update()
                else:
                    print(f'Encountered something weird in {run.name}.')
                    continue
            if model_name not in vae_like_models:
                continue

            dataset = get_field_from_config(run, "dataset")

            if 'mnist' not in dataset:
                continue
            decoder = get_field_from_config(run, "decoder")
            latent_dims = get_field_from_config(run, "latent_dims", type="int")

            prior_flow = get_field_from_config(run, "prior_flow")
            posterior_flow = get_field_from_config(run, "posterior_flow")

            test_ll = get_field_from_summary(run, "test_log_likelihood", type="float")
            test_bpp = get_field_from_summary(run, "test_bpp", type="float")
            test_bpp_adjusted = get_field_from_summary(run, "test_bpp_adjusted", type="float")

            test_loader = get_test_dataloader(dataset)
            posterior_flow = 'none'
            prior_flow = 'none'
            model = get_model(model_name, architecture_size, decoder, latent_dims, image_dim, alpha, posterior_flow,
                              prior_flow)
            #model.loss_function(model.sample(10))  # needed as some components such as actnorm need to be initialized
            run_name = run.name
            artifact = api.artifact(f'nae/{project_name}/{run_name}_best:latest')#run.restore(f'{run_name}_best:latest', run_path=run.path, root='./artifacts')
            artifact_dir = artifact.download()
            artifact_dir = artifact_dir + '/' + os.listdir(artifact_dir)[0]
            model.load_state_dict(torch.load(artifact_dir, map_location=device))
            model = model.to(device)

            test_ll_averager = make_averager()
            for test_batch, _ in test_loader:
                test_batch = dequantize(test_batch)
                test_batch = test_batch.to(device)
                for iw_iter in range(20):
                    log_likelihood = torch.mean(model.approximate_marginal(test_batch, n_samples=128))
                    test_ll_averager(log_likelihood.item())
            new_test_ll = test_ll_averager(None)
            new_bpp_test = bits_per_pixel(new_test_ll, n_pixels)
            new_bpp_test_adjusted = bits_per_pixel(new_test_ll, n_pixels, adjust_value=256.)

            run.summary["test_log_likelihood"] = new_test_ll
            run.summary["test_bpp"] = new_bpp_test
            run.summary['test_bpp_adjusted'] = new_bpp_test_adjusted
            run.summary.update()
            print(f"Updated {run_name}")
        except Exception as e:
            print(e)
            traceback.print_exc()
            print(f'Failed to update {run_name}')
            continue

def add_mse_fid_phase_1():
    use_gpu = True
    device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")
    project_name = 'phase1'
    image_dim = [1, 28, 28]
    n_pixels = np.prod(image_dim)
    vae_like_models = ['vae-iaf', 'vae', 'iwae', 'nae-center', 'nae-corner', 'nae-external']

    alpha = 1e-6

    api = wandb.Api()
    runs = api.runs(path="nae/phase1")
    architecture_size = "small"
    incept = metrics.InceptionV3().to(device)
    for run in runs:
        if run.state != 'finished':
            continue
        if 'fid' in run.summary.keys(): # 'test_rce' in run.summary.keys() and
            if not run.summary['fid'] is None and np.isfinite(run.summary['fid']): #not run.summary['test_rce'] is None and np.isfinite(run.summary['test_rce']) and
                continue

        try:
            model_name = get_field_from_config(run, "model")

            if model_name == 'maf':
                pass

            dataset = get_field_from_config(run, "dataset")

            if 'mnist' not in dataset:
                continue
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

            test_loader = get_test_dataloader(dataset, batch_size=128)
            model = model.eval()

            # if model_name != 'maf' and not 'test_rce' in run.summary.keys() or run.summary['test_rce'] is None or not np.isfinite(run.summary['test_rce']):
            #     with torch.no_grad():
            #         test_rce_averager = make_averager()
            #         for test_batch, _ in test_loader:
            #             test_batch = dequantize(test_batch)
            #             test_batch = test_batch.to(device)
            #
            #             z = model.encode(test_batch)
            #             if isinstance(z, tuple):
            #                 z = z[0]
            #
            #             test_batch_reconstructed = model.decode(z)
            #             if isinstance(test_batch_reconstructed, tuple):
            #                 test_batch_reconstructed = test_batch_reconstructed[0]
            #
            #             rce = torch.mean(F.mse_loss(test_batch_reconstructed, test_batch, reduction='none'))
            #             test_rce_averager(rce.item())
            #
            #     test_rce = test_rce_averager(None)
            #     run.summary["test_rce"] = test_rce
            if not 'fid' in run.summary.keys() or run.summary['fid'] is None or not np.isfinite(run.summary['fid']):
                try:
                    fid = metrics.calculate_fid(model, dataset, device, batch_size=128, incept=incept)
                    run.summary['fid'] = fid
                except Exception as e:
                    print(e)
                    traceback.print_exc()
                    print(f'Failed FID in {run_name}')

            run.summary.update()
            print(f"Updated {run_name}")

        except Exception as e:
            print(e)
            traceback.print_exc()
            print(f'Failed to update {run_name}')
            continue

def add_fid_cifar():
    use_gpu = True
    device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")
    project_name = 'cifar'
    image_dim = [3, 32, 32]

    alpha = 0.05

    api = wandb.Api()
    runs = api.runs(path=f"nae/{project_name}")
    architecture_size = "small"
    incept = metrics.InceptionV3().to(device)
    for run in runs:
        if run.state != 'finished':
            continue

        # if 'fid' in run.summary.keys(): # 'test_rce' in run.summary.keys() and
        #     if not run.summary['fid'] is None and np.isfinite(run.summary['fid']): #not run.summary['test_rce'] is None and np.isfinite(run.summary['test_rce']) and
        #         continue

        try:
            model_name = get_field_from_config(run, "model")
            if 'nae' not in model_name:
                continue
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

            if not 'fid' in run.summary.keys() or run.summary['fid'] is None or not np.isfinite(run.summary['fid']):
                try:
                    fid = metrics.calculate_fid(model, dataset, device, batch_size=128, incept=incept)
                    run.summary['fid'] = fid
                except Exception as e:
                    print(e)
                    traceback.print_exc()
                    print(f'Failed FID in {run_name}')

            run.summary.update()
            print(f"Updated {run_name}")

        except Exception as e:
            print(e)
            traceback.print_exc()
            print(f'Failed to update {run_name}')
            continue

def add_fid_phase2():
    use_gpu = True
    device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")
    project_name = 'phase2'
    image_dim = [3, 32, 32]

    alpha = 0.05

    api = wandb.Api()
    runs = api.runs(path=f"nae/{project_name}")
    incept = metrics.InceptionV3().to(device)
    for run in runs:
        if run.state != 'finished':
            continue
        if 'fid' in run.summary.keys(): # 'test_rce' in run.summary.keys() and
            if not run.summary['fid'] is None and np.isfinite(run.summary['fid']): #not run.summary['test_rce'] is None and np.isfinite(run.summary['test_rce']) and
                continue

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

            if not 'fid' in run.summary.keys() or run.summary['fid'] is None or not np.isfinite(run.summary['fid']):
                try:
                    fid = metrics.calculate_fid(model, dataset, device, batch_size=128, incept=incept, data_dir='celebahq')
                    run.summary['fid'] = fid
                except Exception as e:
                    print(e)
                    traceback.print_exc()
                    print(f'Failed FID in {run_name}')

            run.summary.update()
            print(f"Updated {run_name}")

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



def check_nr_experiments(df):
    datasets = ['mnist', 'cifar', 'fashionmnist', 'kmnist']
    models = ['vae', 'vae-iaf', 'iwae', 'vae-maf-iaf', 'nae-center', 'nae-corner', 'nae-external']
    latent_sizes = [2, 4, 8, 16, 32]

    df.loc[(df.loc[:, 'model'] == 'vae') & (df.loc[:, 'posterior_flow'] == 'iaf') & (df.loc[:, 'prior_flow'] == 'maf'), 'model'] = 'vae-maf-iaf'
    for model in models:
        for dataset in datasets:
            for latent_dims in latent_sizes:
                rows = df.loc[(df.loc[:, 'model'] == model) & (df.loc[:, 'dataset'] == dataset) & (df.loc[:, 'latent_dims'] == latent_dims), :]
                print(f'{model} {dataset} {latent_dims} {rows.shape[0]} rows')
                #print(f'{rows.shape[0]} rows, mean {rows.loc[:, "test_bpp_adjusted"].mean()}, std {rows.loc[:, "test_bpp_adjusted"].std()}')


def check_runs_missing_artifact():
    use_gpu = True
    device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")
    project_name = 'phase1'

    alpha = 1e-6
    image_dim = [1, 28, 28]
    api = wandb.Api()
    runs = api.runs(path="nae/phase1")
    architecture_size = "small"
    for run in runs:
        if run.state != 'finished':
            continue

        try:
            model_name = get_field_from_config(run, "model")

            dataset = get_field_from_config(run, "dataset")

            decoder = get_field_from_config(run, "decoder")
            latent_dims = get_field_from_config(run, "latent_dims", type="int")

            posterior_flow = get_field_from_config('posterior_flow')
            if posterior_flow is None:
                posterior_flow = 'none'
            prior_flow = get_field_from_config('prior_flow')
            if prior_flow is None:
                prior_flow = 'none'
            model = get_model(model_name, architecture_size, decoder, latent_dims, image_dim, alpha, posterior_flow,
                              prior_flow)
            # model.loss_function(model.sample(10))  # needed as some components such as actnorm need to be initialized
            run_name = run.name
            artifact = api.artifact(f'nae/{project_name}/{run_name}_best:latest')
            artifact_dir = artifact.download()
            artifact_dir = artifact_dir + '/' + os.listdir(artifact_dir)[0]
            model.load_state_dict(torch.load(artifact_dir, map_location=device))
        except Exception as e:
            print(e)
            traceback.print_exc()
            print(run.name)

def print_vae_performance(latent_dims):
    n_batches = 16

    model_name = 'vae'

    project_name = 'phase1'
    dataset = 'mnist'
    architecture_size = 'small'
    img_dim = [1, 28, 28]
    alpha = 1e-6

    api = wandb.Api()
    use_gpu = True
    device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")

    runs = api.runs(path=f"nae/{project_name}", filters={
        "config.latent_dims": latent_dims,
        "config.model": model_name,
        "config.dataset": dataset,
        "config.preprocessing": True
    })
    for run_idx, run in enumerate(runs):
        experiment_name = run.name

        decoder = get_field_from_config(run, "decoder")
        latent_dims = get_field_from_config(run, "latent_dims", type="int")

        posterior_flow = get_field_from_config(run, 'posterior_flow')
        prior_flow = get_field_from_config(run, 'prior_flow')

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
        train_dataloader, validation_dataloader, image_dim, alpha = get_train_val_dataloaders(dataset, 128)
        with torch.no_grad():
            batch_loss_printed = False

            approximate_ll_averager = make_averager()
            val_loss_averager = make_averager()
            n_batches_done = 0
            for batch, batch_y in validation_dataloader:

                batch = dequantize(batch)
                batch = batch.to(device)

                batch_loss = torch.mean(model.loss_function(batch))
                val_loss_averager(batch_loss.item())
                for iw_iter in range(20):
                    log_likelihood = torch.mean(model.approximate_marginal(batch, n_samples=128))
                    approximate_ll_averager(log_likelihood.item())

                n_batches_done += 1
                if n_batches_done > n_batches:
                    break

            approximate_ll = approximate_ll_averager(None)
            if not batch_loss_printed:
                print(f'VAE Validation loss: {val_loss_averager(None)}')
                batch_loss_printed = True
            print(f'VAE approximate LL: {approximate_ll}')
        break

def print_aef_center_corner_performance(dataset, latent_dims, n_batches=16):

    model_names = ['aef-center', 'aef-corner']

    project_name = 'phase1'
    dataset = 'mnist'
    architecture_size = 'small'
    img_dim = [1, 28, 28]
    alpha = 1e-6

    api = wandb.Api()
    use_gpu = True
    device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")
    for model_name in model_names:
        runs = api.runs(path=f"nae/{project_name}", filters={
            "config.latent_dims": latent_dims,
            "config.model": model_name,
            "config.dataset": dataset,
            "config.preprocessing": True
        })
        for run_idx, run in enumerate(runs):
            experiment_name = run.name

            decoder = get_field_from_config(run, "decoder")
            latent_dims = get_field_from_config(run, "latent_dims", type="int")

            posterior_flow = get_field_from_config(run, 'posterior_flow')
            prior_flow = get_field_from_config(run, 'prior_flow')

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
            train_dataloader, validation_dataloader, image_dim, alpha = get_train_val_dataloaders(dataset, 128)
            with torch.no_grad():

                val_loss_averager = make_averager()
                n_batches_done = 0
                for batch, batch_y in validation_dataloader:

                    batch = dequantize(batch)
                    batch = batch.to(device)

                    batch_loss = torch.mean(model.loss_function(batch))
                    val_loss_averager(batch_loss.item())

                    n_batches_done += 1
                    if n_batches_done > n_batches:
                        break

                print(f'{model_name} log-likelihood: {val_loss_averager(None)}')
            break



def find_sigma():

    runs_per_latent = 2

    model_name = 'aef-linear'

    # latent_sizes = [256]
    # project_name = 'phase21'
    # dataset = 'celebahq'
    # data_dir = dataset
    # img_dim = [3, 32, 32]
    # alpha = 0.05
    # n_batches = 128
    # batch_size = 16

    latent_sizes = [256]
    project_name = 'phase21'
    dataset = 'imagenet'
    data_dir = 'imagenet'
    #data_dir = 'data/celebahq64'
    img_dim = [3, 32, 32]
    alpha = 0.05
    n_batches = 16 * 4
    iw_batch_size = 32
    val_batch_size = 16
    val_batches = 128

    # latent_sizes = [32]
    # project_name = 'phase1'
    # dataset = 'mnist'
    # data_dir = ""
    # img_dim = [1, 28, 28]
    # alpha = 1e-6
    # n_batches = 16
    # iw_batch_size = 128
    # val_batch_size = 16
    # val_batches = 128


    api = wandb.Api()
    use_gpu = True
    device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")

    #sigmas = [5, 3, 2.5, 2, 1.5, 1, 0.5, 0.1, 0.01, 0.001]
    #sigmas = [0.5, 0.1, 0.075, 0.05, 0.025]
    # sigmas = [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00001]
    # sigmas = [0.05, 0.025, 0.01, 0.005, 0.001, 0.0001]
    #sigmas = [0.1, 0.05, 0.01, 0.075, 0.005, 0.001, 0.00075, 0.0005, 0.00025, 0.0001, 0.00001]
    #sigmas = [0.5, 0.3, 0.2, 0.1, 0.05]
    #sigmas = [20, 10, 5, 2.5, 1, 0.5, 0.25, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00001]
    #sigmas = [0.0005, 0.00025, 0.0001, 0.00005]
    #sigmas = [2, 1, 0.1]#[0.1, 0.001, 0.001]#[5, 2, 1]
    sigmas = [0.00075, 0.00005, 0.00001, 0.000005, 0.000001]
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

            if runs_done_latent == 0:
                runs_done_latent += 1
                continue

            experiment_name = run.name

            print(experiment_name)

            decoder = get_field_from_config(run, "decoder")
            latent_dims = get_field_from_config(run, "latent_dims", type="int")

            posterior_flow = get_field_from_config(run, 'posterior_flow')
            prior_flow = get_field_from_config(run, 'prior_flow')
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

def add_ll_phase2():
    use_gpu = True
    device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")
    #device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")

    latent_sizes = [64]
    project_name = 'phase21'
    dataset = 'celebahq64'
    data_dir = 'celebahq64'
    img_dim = [3, 64, 64]
    alpha = 0.05
    n_batches = 128
    batch_size = 32

    # latent_sizes = [512]
    # project_name = 'phase21'
    # dataset = 'celebahq64'
    # data_dir = 'data/celebahq64'
    # img_dim = [3, 64, 64]
    # alpha = 0.05
    # n_batches = 16 * (128 // 4)
    # batch_size = 4

    n_pixels = np.prod(img_dim)

    api = wandb.Api()
    for latent_dims in latent_sizes:
        test_dataloader = get_test_dataloader(dataset, batch_size, data_dir=data_dir)
        runs = api.runs(path=f"nae/{project_name}", filters={
            "config.model": 'aef-linear',
            "config.dataset": dataset,
            "config.latent_dims": latent_dims
        })
        for run in runs:
            # if 'test_log_likelihood' in run.summary.keys():  # 'test_rce' in run.summary.keys() and
            #     if not run.summary['test_log_likelihood'] is None and np.isfinite(run.summary[
            #                                                           'test_log_likelihood']):  # not run.summary['test_rce'] is None and np.isfinite(run.summary['test_rce']) and
            #         continue
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

                model = get_model(model_name, architecture_size, decoder, latent_dims, img_dim, alpha, posterior_flow,
                                  prior_flow)
                #model.loss_function(model.sample(10))  # needed as some components such as actnorm need to be initialized
                run_name = run.name
                artifact = api.artifact(f'nae/{project_name}/{run_name}_best:latest')
                artifact_dir = artifact.download()
                artifact_dir = artifact_dir + '/' + os.listdir(artifact_dir)[0]
                model.load_state_dict(torch.load(artifact_dir, map_location=device))
                model = model.to(device)
                model.loss_function(model.sample(10))

                model = model.eval()

                importance_std = util.get_posterior_scale_aef_linear(dataset, latent_dims)
                print(f'Approximating LL of {run.name}')
                with torch.no_grad():
                    # Approximate log likelihood if model in VAE family

                    if has_importance_sampling(model):

                        test_ll_averager = make_averager()
                        for test_batch, _ in test_dataloader:
                            test_batch = dequantize(test_batch)
                            test_batch = test_batch.to(device)
                            for iw_iter in range(20):
                                # log_likelihood = torch.mean(model.approximate_marginal(test_batch, n_samples=128))
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
                        print(f"Updated LL of {run_name}")
                    else:
                        print('Something went wrong')

                    if model_name == 'vae':
                        incept = metrics.InceptionV3().to(device)
                        fid = metrics.calculate_fid(model, dataset, device, batch_size=batch_size, incept=incept,
                                                    data_dir=data_dir)
                        run.summary['fid'] = fid
                        run.summary.update()
            except Exception as e:
                print(e)
                traceback.print_exc()
                print(f'Failed to update {run_name}')
                continue


def add_ll_denoising_phase1():
    use_gpu = True
    device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")
    # device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")

    project_name = 'denoising-experiments-5'
    img_dim = [1, 28, 28]
    alpha = 1e-6
    batch_size = 128

    datasets = ['mnist', 'fashionmnist', 'kmnist']

    n_pixels = np.prod(img_dim)

    api = wandb.Api()
    for dataset in datasets:
        test_dataloader = get_test_dataloader(dataset, batch_size)
        runs = api.runs(path=f"nae/{project_name}", filters={
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
                noise_level = get_field_from_config(run, "noise_level", type="float")
                model = get_model(model_name, architecture_size, decoder, latent_dims, img_dim, alpha, posterior_flow,
                                  prior_flow)


                run_name = run.name
                artifact = api.artifact(f'nae/{project_name}/{run_name}_best:latest')
                artifact_dir = artifact.download()
                artifact_dir = artifact_dir + '/' + os.listdir(artifact_dir)[0]
                model.load_state_dict(torch.load(artifact_dir, map_location=device))
                model = model.to(device)
                model.loss_function(model.sample(10))

                model = model.eval()

                importance_std = util.get_posterior_scale_aef_linear(dataset, latent_dims)
                print(f'Approximating LL of {run.name}')
                torch.manual_seed(3)  # Seed noise for equal test comparison
                noise_distribution = torch.distributions.normal.Normal(torch.zeros(batch_size, *img_dim).to(device),
                                                                       noise_level * torch.ones(batch_size,
                                                                                                *img_dim).to(
                                                                           device))
                with torch.no_grad():
                    # Approximate log likelihood if model in VAE family
                    if has_importance_sampling(model):

                        clean_test_ll_averager = make_averager()
                        noisy_test_ll_averager = make_averager()
                        for test_batch, _ in test_dataloader:
                            test_batch = dequantize(test_batch)
                            test_batch = test_batch.to(device)

                            test_batch_noisy = torch.clone(test_batch).detach()
                            test_batch_noisy += noise_distribution.sample()[:test_batch.shape[0]]
                            test_batch_noisy = torch.clamp(test_batch_noisy, 0., 1.)
                            test_batch_noisy = test_batch_noisy.to(device)

                            for iw_iter in range(20):
                                if model_name == 'aef-linear':
                                    log_likelihood_clean = torch.mean(
                                        model.approximate_marginal(test_batch, n_samples=128, std=importance_std))
                                    log_likelihood_noisy = torch.mean(
                                        model.approximate_marginal(test_batch_noisy, n_samples=128, std=importance_std))
                                else:
                                    log_likelihood_clean = torch.mean(model.approximate_marginal(test_batch,
                                                                                                 n_samples=128))
                                    log_likelihood_noisy = torch.mean(
                                        model.approximate_marginal(test_batch_noisy, n_samples=128))

                                clean_test_ll_averager(log_likelihood_clean.item())
                                noisy_test_ll_averager(log_likelihood_noisy.item())
                        test_ll_clean = clean_test_ll_averager(None)
                        test_ll_noisy = noisy_test_ll_averager(None)


                        bpp_test_adjusted_clean = bits_per_pixel(test_ll_clean, n_pixels, adjust_value=256.)
                        bpp_test_adjusted_noisy = bits_per_pixel(test_ll_noisy, n_pixels, adjust_value=256.)

                        run.summary['test_log_likelihood_clean'] = test_ll_clean
                        run.summary['test_bpp_adjusted_clean'] = bpp_test_adjusted_clean
                        run.summary['test_bpp_adjusted_noisy'] = bpp_test_adjusted_noisy
                        run.summary.update()
                        print(f"Updated {run_name}")
                    else:
                        print('Something went wrong')

            except Exception as e:
                print(e)
                traceback.print_exc()
                print(f'Failed to update {run_name}')
                continue

def add_ll_imagenet():
    use_gpu = True
    device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")

    project_name = 'phase21'
    dataset = 'imagenet'
    data_dir = 'imagenet'
    img_dim = [3, 32, 32]
    alpha = 0.05
    batch_size = 16

    n_pixels = np.prod(img_dim)

    to_do = ['aef-linear_big_imagenet_latent_size_256_post_maf_prior_maf_4JtW_continued',
             'aef-linear_big_imagenet_latent_size_128_post_maf_prior_maf_TNWh_continued',
             'aef-linear_big_imagenet_latent_size_256_post_maf_prior_maf_wzhK_continued',
             'aef-linear_big_imagenet_latent_size_128_post_maf_prior_maf_IwmN_continued',
             'vae_big_imagenet_latent_size_256_post_iaf_prior_maf_6ghG_continued']

    api = wandb.Api()
    test_dataloader = get_test_dataloader(dataset, batch_size, data_dir=data_dir)
    runs = api.runs(path=f"nae/{project_name}", filters={
        "config.dataset": dataset,
    })
    for run in runs:
        if run.name not in to_do:
            continue
        try:
            model_name = get_field_from_config(run, "model")
            dataset = get_field_from_config(run, "dataset")
            decoder = get_field_from_config(run, "decoder")
            latent_dims = get_field_from_config(run, "latent_dims", type="int")
            architecture_size = get_field_from_config(run, "architecture_size")

            posterior_flow = get_field_from_config(run, "posterior_flow")
            prior_flow = get_field_from_config(run, "prior_flow")

            model = get_model(model_name, architecture_size, decoder, latent_dims, img_dim, alpha, posterior_flow,
                              prior_flow)
            run_name = run.name
            artifact = api.artifact(f'nae/{project_name}/{run_name}_best:latest')
            artifact_dir = artifact.download()
            artifact_dir = artifact_dir + '/' + os.listdir(artifact_dir)[0]
            model.load_state_dict(torch.load(artifact_dir, map_location=device))
            model = model.to(device)
            model.loss_function(model.sample(10))

            model = model.eval()

            if model_name == 'aef-linear':
                importance_std = util.get_posterior_scale_aef_linear(dataset, latent_dims)
            print(f'Approximating LL of {run.name}')
            with torch.no_grad():
                # Approximate log likelihood if model in VAE family

                if has_importance_sampling(model):

                    test_ll_averager = make_averager()
                    for test_batch, _ in test_dataloader:
                        test_batch = dequantize(test_batch)
                        test_batch = test_batch.to(device)
                        for iw_iter in range(20):
                            if model_name == 'vae':
                                log_likelihood = torch.mean(model.approximate_marginal(test_batch, n_samples=128))
                            else:
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
    add_ll_imagenet()
    # find_sigma()
    #add_ll_phase2()
    #add_ll_phase1()
    # add_ife()
    #find_optimal_fid_celeba(False)
    #update_nae_external()
    # add_fid_cifar()
    # add_mse_fid_phase_1()
    # df = extract_data_from_runs('denoising-experiments-5')
    # df.to_pickle('denoising-experiments-5.pkl')
    #df = pd.read_pickle('denoising-experiments-5.pkl')
    #generate_denoising_table(df)
    # generate_phase1_table(df, 2)
    # generate_phase1_table(df, 32)
    # phase1_fid_plot(df)
    # check_nr_experiments(df)
    # check_runs_missing_artifact()
    # df = extract_data_from_runs('phase21')
    # df.to_pickle('phase21.pkl')
    # df = pd.read_pickle('phase1.pkl')
    # generate_phase2_table(df, 64)
    # generate_phase2_table(df, 128)
    # generate_phase2_table(df, 256)


    # generate_denoising_table(df)
    #denoising_plot(df)
    # df = pd.read_pickle('phase1.pkl')
    # phase1_bpp_plot(df)

    # api = wandb.Api()
    # runs = api.runs(path="nae/phase1")
    #
    # for run in runs:
    #     print(run.name)
    # exit()

    #add_mse_fid_phase_1()



    exit()