import os
import traceback

import numpy as np
import torch
import torch.nn.functional as F

import metrics
import wandb
import pandas as pd

from models.model_database import get_model
from util import load_best_model, vae_log_prob, make_averager, dequantize, bits_per_pixel
from datasets import get_test_dataloader
import matplotlib.pyplot as plt



def generate_loss_over_latentdims():
    api = wandb.Api()

    project_name = 'phase1'
    image_dim = [1, 28, 28]

    alpha = 1e-6
    use_center_pixels = False
    use_gpu = False
    device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")

    dataset = 'mnist'
    model_name = 'nae'
    latent_sizes = [2, 4, 8, 16, 32]
    n_runs = 5
    decoder = 'independent'
    nae_type = 'corner'

    scores = np.zeros((n_runs, len(latent_sizes)))
    # n_recorded = np.zeros((len(latent_sizes)))
    for idx, latent_size in enumerate(latent_sizes):
        runs = api.runs(path="nae/phase1", filters={"config.dataset": dataset,
                                                    "config.latent_dim": latent_size,
                                                    "config.model": model_name,
                                                    "config.decoder": decoder,
                                                    })
        n_recorded = 0
        for run in runs:
            if nae_type not in run.name:
                continue
            if n_recorded >= n_runs:
                print(f'Found more than {n_runs} with latent size {latent_size}')
                break
            scores[n_recorded, idx] = run.summary['test_loss']
            n_recorded += 1
    scores[scores == np.inf] = np.nan
    best_scores = -1 * np.nanmin(scores, axis=0)  # TODO: change to mean once inf/positive runs are gone
    fig = plt.figure()
    plt.scatter(np.arange(len(latent_sizes)), best_scores)
    plt.ylabel('Log-likelihood')
    plt.xlabel('Dimensionality of latent space')
    plt.xticks(np.arange(len(latent_sizes)), labels=[f'$2^{i + 1}$' for i in range(len(latent_sizes))])

    # GET MAF likelihood
    scores_maf = np.zeros((n_runs,))
    runs = api.runs(path="nae/phase1", filters={"config.dataset": dataset,
                                                "config.model": 'maf',
                                                })
    n_recorded = 0
    for run in runs:
        if n_recorded >= n_runs:
            print(f'Found more than {n_runs} for maf')
            break
        scores_maf[n_recorded] = run.summary['test_loss']
        n_recorded += 1
    plt.axhline(y=-1 * np.min(scores_maf), color='k', linestyle='--')

    plt.savefig('./plots/likelihood_for_latents.png', dpi='figure')

def extract_data_from_runs(project_name='phase1'):
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


    runs = api.runs(path=f"nae/{project_name}")
    for run in runs:
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
        fid.append(get_field_from_summary(run, "test_fid", type="float"))
        noise_level.append(get_field_from_summary(run, "noise_level", type="float"))

        model_name = get_field_from_config(run, "model")
        model_names.append(model_name)

        run_nr_idx = run.name.find('run_')
        if run_nr_idx != -1:
            run_nr.append(int(run.name[run_nr_idx + 4]))
        else:
            run_nr.append(None)

    col_dict = {'model': model_names,
                'dataset': dataset,
                'latent_dim': latent_dims,
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
                'noise_level': noise_level
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
    if field not in run.summary.keys():
        return None
    if type == 'int':
        return int(run.summary[field])
    elif type == 'float':
        return float(run.summary[field])
    else:
        return run.summary[field]


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

        try:
            model_name = get_field_from_config(run, "model")

            dataset = get_field_from_config(run, "dataset")

            if 'mnist' not in dataset:
                continue
            decoder = get_field_from_config(run, "decoder")
            latent_dims = get_field_from_config(run, "latent_dims", type="int")


            posterior_flow = 'none'
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
            with torch.no_grad():
                test_rce_averager = make_averager()
                for test_batch, _ in test_loader:
                    test_batch = dequantize(test_batch)
                    test_batch = test_batch.to(device)

                    z = model.encode(test_batch)
                    if isinstance(z, tuple):
                        z = z[0]

                    test_batch_reconstructed = model.decode(z)
                    if isinstance(test_batch_reconstructed, tuple):
                        test_batch_reconstructed = test_batch_reconstructed[0]

                    rce = torch.mean(F.mse_loss(test_batch_reconstructed, test_batch, reduction='none'))
                    test_rce_averager(rce.item())

            test_rce = test_rce_averager(None)
            run.summary["test_rce"] = test_rce
            #
            fid = metrics.calculate_fid(model, dataset, device, batch_size=128, incept=incept)
            run.summary['fid'] = fid

            run.summary.update()
            print(f"Updated {run_name} with RCE and FID")

        except Exception as e:
            print(e)
            traceback.print_exc()
            print(f'Failed to update {run_name}')
            continue

def generate_phase1_table(latent_dims=32):

    datasets = ['mnist', 'fashionmnist', 'kmnist']
    models = ['vae', 'iwae', 'vae-iaf', 'nae-center', 'nae-corner', 'nae-external']
    model_titles = ['VAE', 'IWAE', 'VAE-IAF', 'IAE (center)', 'IAE (corner)', 'IAE (linear)']

    df = extract_data_from_runs(project_name='denoising-experiments')

    mean_bpps = np.zeros((len(models), len(datasets)))
    se_bpps = np.zeros((len(models), len(datasets)))
    for model_idx, model_name in enumerate(models):
        for dataset_idx, dataset in enumerate(datasets):
            runs = df.loc[(df.loc[:, 'model'] == model_name) & (df.loc[:, 'dataset'] == dataset) & (df.loc[:, 'latent_dim'] == latent_dims)]
            mean_bpps[model_idx, dataset_idx] = runs.loc[:, 'test_bpp_adjusted'].mean(axis=0)
            se_bpps[model_idx, dataset_idx] = runs.loc[:, 'test_bpp_adjusted'].sem(axis=0)

    for row_idx in range(len(models)):
        print(model_titles[row_idx], end=' ')
        for col_idx in range(len(datasets)):
            print(f'& ${mean_bpps[row_idx, col_idx]:.3f} \pm {se_bpps[row_idx, col_idx]:.3f}$', end=' ')
        print('\\\\')

    print('MAF', end=' ')
    for dataset_idx, dataset in enumerate(datasets):
        runs = df.loc[(df.loc[:, 'model'] == 'maf') & (df.loc[:, 'dataset'] == dataset)]
        mean_maf = runs.loc[:, 'test_bpp_adjusted'].mean(axis=0)
        se_maf = runs.loc[:, 'test_bpp_adjusted'].sem(axis=0)
        print(f'& ${mean_maf:.3f} \pm {se_maf:.3f}$', end=' ')

def generate_denoising_table(latent_dims=32):

    datasets = ['mnist', 'fashionmnist']
    models = ['ae', 'vae', 'vae-iaf', 'nae-external']
    model_titles = ['AE', 'VAE', 'VAE-IAF', 'IAE (linear)']
    noise_levels = [0.25, 0.5, 0.75]

    df = extract_data_from_runs('denoising-experiments')

    mean_rce = np.zeros((len(models), len(datasets), len(noise_levels)))
    se_rce = np.zeros((len(models), len(datasets), len(noise_levels)))
    for model_idx, model_name in enumerate(models):
        for dataset_idx, dataset in enumerate(datasets):
            for noise_idx, noise_level in enumerate(noise_levels):
                runs = df.loc[(df.loc[:, 'model'] == model_name) & (df.loc[:, 'dataset'] == dataset) & (df.loc[:, 'noise_level'] == noise_level)]
                print(f'{model_name} {dataset} {noise_level} nr. of runs: {len(runs)}')
                mean_rce[model_idx, dataset_idx, noise_idx] = runs.loc[:, 'test_rce_with_noise'].mean(axis=0)
                se_rce[model_idx, dataset_idx] = runs.loc[:, 'test_rce_with_noise'].sem(axis=0)

    for row_idx in range(len(models)):
        print(model_titles[row_idx], end=' ')
        for dataset_idx in range(len(datasets)):
            for noise_level_idx in range(len(noise_levels)):
                print(f'& ${mean_rce[row_idx, dataset_idx, noise_level_idx]:.3f} \pm {se_rce[row_idx, dataset_idx, noise_level_idx]:.3f}$', end=' ')
        print('\\\\')



if __name__ == '__main__':
    # df = extract_data_from_runs()
    # print(df)
    #generate_denoising_table()
    add_mse_fid_phase_1()

