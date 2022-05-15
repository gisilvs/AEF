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
import seaborn as sns

from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)


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
        if run.state != 'finished':
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
        fid.append(get_field_from_summary(run, "test_fid", type="float"))
        noise_level.append(get_field_from_config(run, "noise_level", type="float"))

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
    if field not in run.summary.keys() or run.summary[field] is None:
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
            runs = df.loc[(df.loc[:, 'model'] == model_name) & (df.loc[:, 'dataset'] == dataset) & (df.loc[:, 'latent_dims'] == latent_dims)]
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

def generate_denoising_table(df, latent_dims=32):

    datasets = ['mnist', 'kmnist']
    models = ['ae', 'vae-iaf', 'nae-external']
    model_titles = ['AE', 'VAE-IAF', 'IAE (linear)']
    noise_levels = [0.25, 0.5, 0.75]

    mean_rce = np.zeros((len(models), len(datasets), len(noise_levels)))
    se_rce = np.zeros((len(models), len(datasets), len(noise_levels)))
    for model_idx, model_name in enumerate(models):
        for dataset_idx, dataset in enumerate(datasets):
            for noise_idx, noise_level in enumerate(noise_levels):
                runs = df.loc[(df.loc[:, 'model'] == model_name) & (df.loc[:, 'dataset'] == dataset) & (df.loc[:, 'noise_level'] == noise_level)]
                #print(f'{model_name} {dataset} {noise_level} nr. of runs: {len(runs)}')
                mean_rce[model_idx, dataset_idx, noise_idx] = runs.loc[:, 'test_rce_with_noise'].mean(axis=0)
                se_rce[model_idx, dataset_idx] = runs.loc[:, 'test_rce_with_noise'].sem(axis=0)

    for row_idx in range(len(models)):
        print(model_titles[row_idx], end=' ')
        for dataset_idx in range(len(datasets)):
            for noise_level_idx in range(len(noise_levels)):
                print(f'& ${mean_rce[row_idx, dataset_idx, noise_level_idx]:.3f} \pm {se_rce[row_idx, dataset_idx, noise_level_idx]:.3f}$', end=' ')
        print('\\\\')



def denoising_plot(df):
    datasets = ['mnist', 'kmnist', 'fashionmnist']
    dataset_titles = {'mnist':'MNIST', 'kmnist': 'KMNIST', 'fashionmnist' : 'FashionMNIST'}
    models = ['ae', 'vae-iaf', 'nae-external']
    model_titles = ['AE', 'VAE-IAF', 'IAE (linear)']
    noise_levels = [0.25, 0.5, 0.75]


    # for dataset_idx, dataset in enumerate(datasets):
    #     mean_rce = np.zeros((len(models), len(noise_levels)))
    #     se_rce = np.zeros((len(models), len(noise_levels)))
    #     for model_idx, model_name in enumerate(models):
    #             for noise_idx, noise_level in enumerate(noise_levels):
    #                 runs = df.loc[(df.loc[:, 'model'] == model_name) & (df.loc[:, 'dataset'] == dataset) & (
    #                             df.loc[:, 'noise_level'] == noise_level)]
    #                 # print(f'{model_name} {dataset} {noise_level} nr. of runs: {len(runs)}')
    #                 mean_rce[model_idx, dataset_idx, noise_idx] = runs.loc[:, 'test_rce_with_noise'].mean(axis=0)
    #                 se_rce[model_idx, dataset_idx] = 1.96 * runs.loc[:, 'test_rce_with_noise'].sem(axis=0)
    # Replace values

    plt.rcParams['axes.axisbelow'] = True

    for dataset in datasets:
        df_to_use = df.loc[df.loc[:, 'dataset'] == dataset]
        df_to_use = df_to_use.replace(to_replace={'ae': "AE", 'vae-iaf': "VAE-IAF", 'nae-external': 'IAE'})
        df_to_use = df_to_use.sort_values(by=['model'])

        ax = sns.pointplot(x="noise_level", y="test_rce_with_noise", hue="model", data=df_to_use, ci=95)
        ax.set_facecolor('lavender')
        ax.yaxis.set_major_locator(plt.MaxNLocator(4))
        ax.grid(visible=True, which='major', axis='both', color='w')


        ax.legend(ncol=3)
        ax.set_xlabel('Noise level')
        ax.set_ylabel('Avg. reconstruction error')
        plt.title(dataset_titles[dataset])
        plt.savefig(f'plots/denoising_{dataset}.png')

def phase1_plot(df, broken_axis=True):
    datasets = ['mnist', 'kmnist', 'fashionmnist']
    models = ['ae', 'vae-iaf', 'nae-external']
    dataset_titles = {'mnist': 'MNIST', 'kmnist': 'KMNIST', 'fashionmnist': 'FashionMNIST'}

    noise_levels = [0.25, 0.5, 0.75]

    vae_models = ['vae', 'iwae', 'vae-iaf']
    nae_models = ['nae-external', 'nae-corner', 'nae-center']

    # Replace names

    df_fixed = df.loc[(df.loc[:,'latent_dims'] <= 32) & (df.loc[:, 'model'] != 'maf'), :]

    # todo
    # rc = {
    #     "text.usetex": True,
    #     "font.family": "Times New Roman",
    #     }
    # plt.rcParams.update(rc)

    for dataset in datasets:
        df_to_use = df_fixed[df.loc[:, 'dataset'] == dataset]
        #df_to_use = df_to_use[df.loc[:, 'model'].isin(['vae', 'vae-iaf', 'iwae'])]
        df_to_use = df_to_use.replace(to_replace={'iwae': "vae-iwae"}) #hack
        df_to_use = df_to_use.sort_values(by=['model'])
        df_to_use = df_to_use.replace(to_replace={'vae-iwae': "iwae"})
        df_to_use = df_to_use.replace(to_replace={'vae': "VAE",
                                     'iwae': "IWAE",
                                     'vae-iaf': "VAE-IAF",
                                     'nae-center': 'IAE (center)',
                                     'nae-corner': 'IAE (corner)',
                                     'nae-external': 'IAE (linear)'})

        if not broken_axis:
            ax = sns.pointplot(x="latent_dims", y="test_bpp_adjusted", hue="model", data=df_to_use, ci=95)
        else:

            top_scores = df_to_use[df.loc[:, 'model'].isin(vae_models)].loc[:, 'test_bpp_adjusted']
            bottom_scores = df_to_use[df.loc[:, 'model'].isin(nae_models)].loc[:, 'test_bpp_adjusted']
            max_top, min_top = top_scores.max(), top_scores.min()
            top_range = max_top - min_top
            max_bottom, min_bottom = bottom_scores.max(), bottom_scores.min()
            bottom_range = max_bottom - min_bottom


            fig, (ax_top, ax_bottom) = plt.subplots(2, 1, sharex=True, dpi=300, figsize=(6,6))
            fig.subplots_adjust(hspace=0.05)  # adjust space between axes

            sns.pointplot(x="latent_dims", y="test_bpp_adjusted", hue="model", data=df_to_use, ci=95, ax=ax_top)
            # ax_top.xaxis.set_major_locator(MultipleLocator(top_range/6))
            # ax_top.yaxis.set_major_locator(MultipleLocator(top_range/6))

            #sns.set_theme()
            sns.pointplot(x="latent_dims", y="test_bpp_adjusted", hue="model", data=df_to_use, ci=95, ax=ax_bottom)

            

            ax_top.grid(visible=True, which='major', axis='both', color='w')
            ax_bottom.grid(visible=True, which='major', axis='both', color='w')
            #sns.set_theme()
            ax_top.set_facecolor('lavender')
            ax_bottom.set_facecolor('lavender')
            ax_top.yaxis.set_major_locator(plt.MaxNLocator(4))
            ax_bottom.yaxis.set_major_locator(plt.MaxNLocator(4))



            ax_top.set_ylim(min_top - 0.2 * top_range, max_top + 0.5 * top_range)
            ax_bottom.set_ylim(min_bottom - 0.2 * bottom_range, max_bottom + 0.2 * bottom_range)

            sns.despine(ax=ax_bottom)
            sns.despine(ax=ax_top, bottom=True)


            ax = ax_top
            d = .015  # how big to make the diagonal lines in axes coordinates
            # arguments to pass to plot, just so we don't keep repeating them
            kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
            ax.plot((-d, +d), (-d, +d), **kwargs)  # top-left diagonal

            ax2 = ax_bottom
            kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
            ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal

            # remove one of the legend


            ax_top.tick_params(bottom=False)

            #ax_bottom.set_xlabel('Nr. of latent dimensions')
            ax_top.set_xlabel('')
            ax_top.set_ylabel('')
            ax_bottom.set_xlabel('')
            ax_bottom.set_ylabel('')
            #ax.set_ylabel('Bits per pixel')
            fig.add_subplot(111, frameon=False)
            plt.xlabel("Nr. of latent dimensions")
            # hide tick and tick label of the big axis
            plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)

            #ax_bottom.set_xlabel("Nr. of latent dimensions")
            #plt.ylabel("Bits per pixel")
            fig.text(0.005, 0.5, 'Bits per pixel', va='center', rotation='vertical')

            ax_top.legend(loc='upper center', ncol=3)#, bbox_to_anchor=(0.5, -0.1))
            ax_bottom.legend_.remove()

            #fig.subplots_adjust(bottom=0.2)
            # handles = ax_top.legend_.data.values()
            # labels = ax_top.legend_.data.keys()
            #
            # ax_bottom.legend(handles=handles, labels=labels, loc='lower center', ncol=6)

            # Shrink current axis's height by 10% on the bottom
            # box = ax.get_position()
            # ax.set_position([box.x0, box.y0 + box.height * 0.1,
            #                  box.width, box.height * 0.9])

            ax_top.set_title(dataset_titles[dataset])
        plt.savefig(f'plots/phase1_{dataset}.png')

def check_nr_experiments(df):
    datasets = ['mnist', 'cifar', 'fashionmnist', 'kmnist']
    models = ['vae', 'vae-iaf', 'iwae', 'vae-maf-iaf', 'nae-center', 'nae-corner', 'nae-external']
    latent_sizes = [2, 4, 8, 16, 32]

    df.loc[(df.loc[:, 'model'] == 'vae') & (df.loc[:, 'posterior_flow'] == 'iaf') & (df.loc[:, 'prior_flow'] == 'maf'), 'model'] = 'vae-maf-iaf'
    for model in models:
        for dataset in datasets:
            for latent_dims in latent_sizes:
                rows = df.loc[(df.loc[:, 'model'] == model) & (df.loc[:, 'dataset'] == dataset) & (df.loc[:, 'latent_dims'] == latent_dims), :]
                print(f'{rows.shape[0]} rows, mean {rows.loc[:, "test_bpp_adjusted"].mean()}, std {rows.loc[:, "test_bpp_adjusted"].std()}')


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


if __name__ == '__main__':
    # df = extract_data_from_runs('phase1')
    # df.to_pickle('phase1.pkl')
    # df = pd.read_pickle('phase1.pkl')
    # check_nr_experiments(df)
    check_runs_missing_artifact()
    # df = extract_data_from_runs('denoising-experiments-1')
    # df.to_pickle('denoising-experiments-1.pkl')
    # df = pd.read_pickle('denoising-experiments-1.pkl')
    # denoising_plot(df)
    # df = pd.read_pickle('phase1.pkl')
    # phase1_plot(df)

    # api = wandb.Api()
    # runs = api.runs(path="nae/phase1")
    #
    # for run in runs:
    #     print(run.name)
    # exit()
    #add_mse_fid_phase_1()

