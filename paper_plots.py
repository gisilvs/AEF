import os
import sys
import traceback

import pandas as pd
import torch
import torchvision

import wandb
from matplotlib import pyplot as plt
import seaborn as sns
from tueplots import bundles

from analysis import get_field_from_config, extract_data_from_runs
from datasets import get_test_dataloader, get_train_val_dataloaders
from models import model_database
from models.model_database import get_model
from util import plot_image_grid
from visualize import plot_reconstructions, plot_noisy_reconstructions, plot_samples, plot_latent_space_2d, get_z_values

def phase1_old_bpp_fid(df):
    datasets = ['mnist', 'kmnist', 'fashionmnist']
    models_full = ['vae', 'nae-external', 'nae-corner', 'nae-center']
    models_main = ['vae', 'nae-external']
    dataset_titles = {'mnist': 'MNIST', 'kmnist': 'KMNIST', 'fashionmnist': 'FashionMNIST'}

    # todo
    rc = {
        "text.usetex": True,
        "font.family": "Times New Roman",
        "axes.axisbelow": True,
    }
    plt.rcParams.update(rc)

    for dataset in datasets:
        df_to_use = df[(df.loc[:, 'dataset'] == dataset) & (df.loc[:, 'model'].isin(models_main))]
        df_to_use = df_to_use.sort_values(by='model', axis=0)
        df_to_use = df_to_use.replace(to_replace={'vae': "VAE",
                                                  'nae-center': 'AEF (center)',
                                                  'nae-corner': 'AEF (corner)',
                                                  'nae-external': 'AEF (linear)'})

        fig = plt.figure(dpi=300, figsize=(6, 6))
        ax = sns.pointplot(x="latent_dims", y="test_bpp_adjusted", hue="model", data=df_to_use, ci=95)
        ax.set_facecolor('lavender')
        # ax.yaxis.set_major_locator(plt.MaxNLocator(4))
        ax.grid(visible=True, which='major', axis='both', color='w')

        ax.legend(ncol=2, loc='upper center')
        ax.set_xlabel('Latent dimensions')
        ax.set_ylabel('BPD')
        plt.title(dataset_titles[dataset])
        plt.savefig(f'plots/iclr/bpp_{dataset}_old_main.pdf', bbox_inches='tight')

        df_to_use = df[(df.loc[:, 'dataset'] == dataset) & (df.loc[:, 'model'].isin(models_full))]
        df_to_use = df_to_use.sort_values(by='model', axis=0)
        df_to_use = df_to_use.replace(to_replace={'vae': "VAE",
                                                  'nae-center': 'AEF (center)',
                                                  'nae-corner': 'AEF (corner)',
                                                  'nae-external': 'AEF (linear)'})

        fig = plt.figure(dpi=300, figsize=(6, 3))
        ax = sns.pointplot(x="latent_dims", y="test_bpp_adjusted", hue="model", data=df_to_use, ci=95)
        bottom, top = plt.ylim()  # return the current ylim
        plt.ylim((bottom, top + 0.1))  # set the ylim to bottom, top

        ax.set_facecolor('lavender')
        # ax.yaxis.set_major_locator(plt.MaxNLocator(4))
        ax.grid(visible=True, which='major', axis='both', color='w')

        ax.legend(ncol=2, loc='upper center')
        ax.set_xlabel('Latent dimensions')
        ax.set_ylabel('BPD')
        plt.title(dataset_titles[dataset])
        plt.savefig(f'plots/iclr/bpp_{dataset}_old_full.pdf', bbox_inches='tight')

        fig = plt.figure(dpi=300, figsize=(6, 3))
        ax = sns.pointplot(x="latent_dims", y="fid", hue="model", data=df_to_use, ci=95)
        ax.set_facecolor('lavender')
        # ax.yaxis.set_major_locator(plt.MaxNLocator(4))
        ax.grid(visible=True, which='major', axis='both', color='w')

        ax.legend(ncol=2, loc='upper center')
        ax.set_xlabel('Latent dimensions')
        ax.set_ylabel('FID')
        plt.title(dataset_titles[dataset])
        plt.savefig(f'plots/iclr/fid_{dataset}_old.pdf', bbox_inches='tight')


def generate_denoising_reconstructions_main():
    dataset = 'celebahq'
    models = ['ae', 'vae', 'aef-linear']

    latent_dims = 128
    api = wandb.Api()
    img_dim = [3, 32, 32]
    alpha = 0.05
    project_name = 'denoising-experiments-6'
    noise_level = 0.1
    architecture_size = 'big'

    use_gpu = True
    device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")

    noise_distribution = torch.distributions.normal.Normal(torch.zeros([60, *img_dim]),
                                                           noise_level * torch.ones([60, *img_dim]))
    test_loader = get_test_dataloader(dataset, data_dir='celebahq')


    for model_name in models:
        runs = api.runs(path=f"nae/{project_name}", filters={"config.dataset": dataset,
                                                             "config.latent_dims": latent_dims,
                                                             "config.model": model_name,
                                                             "config.noise_level": noise_level
                                                             })

        for run in runs:
            experiment_name = run.name
            try:
                decoder = get_field_from_config(run, 'decoder')
                posterior_flow = get_field_from_config(run, 'posterior_flow')
                if posterior_flow is None:
                    posterior_flow = 'none'
                prior_flow = get_field_from_config(run, 'prior_flow')
                if prior_flow is None:
                    prior_flow = 'none'

                model = model_database.get_model(model_name, architecture_size, decoder, latent_dims, img_dim,
                                                 alpha, posterior_flow, prior_flow)

                run_name = run.name
                artifact = api.artifact(
                    f'nae/{project_name}/{run_name}_best:latest')  # run.restore(f'{run_name}_best:latest', run_path=run.path, root='./artifacts')
                artifact_dir = artifact.download()
                artifact_dir = artifact_dir + '/' + os.listdir(artifact_dir)[0]
                model.load_state_dict(torch.load(artifact_dir, map_location=device))
                model = model.to(device)

                batch_iter = iter(test_loader)

                torch.manual_seed(3)
                for i in range(20):
                    image_batch = next(batch_iter)[0]
                    img = plot_noisy_reconstructions(model, image_batch, device, noise_distribution,
                                                     img_dim, n_rows=3, n_cols=6)
                    img.save(f'denoising_main2/{run_name}_{i}.png')
            except Exception as E:
                print(E)
                print(f'Failed to plot latent space of {experiment_name}')
                traceback.print_exc()
                continue

def generate_denoising_reconstructions_supp():
    dataset = 'celebahq'
    models = ['ae', 'vae', 'aef-linear']

    latent_dims = 128
    api = wandb.Api()
    img_dim = [3, 32, 32]
    alpha = 0.05
    project_name = 'denoising-experiments-2'
    noise_levels = [0.05, 0.1, 0.2]
    architecture_size = 'big'

    use_gpu = True
    device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")

    for noise_level in noise_levels:
        noise_distribution = torch.distributions.normal.Normal(torch.zeros([60, *img_dim]),
                                                               noise_level * torch.ones([60, *img_dim]))
        test_loader = get_test_dataloader(dataset, data_dir='celebahq')
        batch_iter = iter(test_loader)

        for i in range(12):
            image_batch = next(batch_iter)[0]
        for model_name in models:
            runs = api.runs(path=f"nae/{project_name}", filters={"config.dataset": dataset,
                                                                 "config.latent_dims": latent_dims,
                                                                 "config.model": model_name,
                                                                 "config.noise_level": noise_level
                                                                 })

            for run in runs:
                run_id = run.id
                experiment_name = run.name
                try:
                    decoder = get_field_from_config(run, 'decoder')
                    posterior_flow = get_field_from_config(run, 'posterior_flow')
                    if posterior_flow is None:
                        posterior_flow = 'none'
                    prior_flow = get_field_from_config(run, 'prior_flow')
                    if prior_flow is None:
                        prior_flow = 'none'

                    model = model_database.get_model(model_name, architecture_size, decoder, latent_dims, img_dim,
                                                     alpha, posterior_flow, prior_flow)

                    run_name = run.name
                    artifact = api.artifact(
                        f'nae/{project_name}/{run_name}_best:latest')  # run.restore(f'{run_name}_best:latest', run_path=run.path, root='./artifacts')
                    artifact_dir = artifact.download()
                    artifact_dir = artifact_dir + '/' + os.listdir(artifact_dir)[0]
                    model.load_state_dict(torch.load(artifact_dir, map_location=device))
                    model = model.to(device)


                    torch.manual_seed(3)

                    grid = plot_noisy_reconstructions(model, image_batch, device, noise_distribution,
                                                      img_dim, n_rows=6, n_cols=6)
                    img = torchvision.transforms.ToPILImage()(grid)
                    img.save(f'denoising_celeb/{run_name}.png')
                    break
                except Exception as E:
                    print(E)
                    print(f'Failed to plot latent space of {experiment_name}')
                    traceback.print_exc()
                    continue


def generate_plots_abstract():
    dataset = 'celebahq64'

    latent_sizes = [128, 256, 512]
    api = wandb.Api()
    img_dim = [3, 64, 64]
    alpha = 0.05
    project_name = 'phase21'

    architecture_size = 'big'

    use_gpu = True
    device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")

    test_loader = get_test_dataloader(dataset, data_dir=dataset, batch_size=6)
    for latent_dims in latent_sizes:
        runs = api.runs(path=f"nae/{project_name}", filters={
            "config.latent_dims": latent_dims,
            "config.dataset": dataset
            #"config.preprocessing": True,
        })
        for run_idx, run in enumerate(runs):
            run_id = run.id
            experiment_name = run.name

            model_name = get_field_from_config(run, "model")

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

            for i in range(20):
                for temp in [0.8, 0.85, 0.9, 1.0]:
                    img = plot_samples(model, img_dim, n_rows=2, n_cols=3, batch_size=64, temperature=temp, padding=0)
                    img.save(f'samples_abstract/{run_name}_{temp}_{i}.png')

            for i in range(50):
                img = plot_reconstructions(model, test_loader, device, img_dim, n_rows=2, n_cols=3,
                                            skip_batches=i, padding=0)
                img.save(f'reconstruction_abstract/{run_name}_{i}.png')

def generate_celeba_samples_main():
    model_names = ['vae', 'aef-linear']
    latent_sizes = [64]

    project_name = 'phase21'

    architecture_size = 'big'
    dataset = 'celebahq64'
    img_dim = [3, 64, 64]
    alpha = 0.05

    api = wandb.Api()
    use_gpu = True
    device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")

    for model_idx, model_name in enumerate(model_names):
        for latent_idx, latent_dims in enumerate(latent_sizes):
            runs = api.runs(path=f"nae/{project_name}", filters={
                "config.latent_dims": latent_dims,
                "config.model": model_name,
                "config.dataset": dataset
            })
            for run_idx, run in enumerate(runs):
                run_id = run.id
                experiment_name = run.name

                model_name = get_field_from_config(run, "model")

                dataset = get_field_from_config(run, "dataset")

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

                for i in range(10):
                    img = plot_samples(model, img_dim, n_rows=2, n_cols=2, batch_size=64, padding=0, temperature=0.85)

                    img.save(f'celeba_samples2/{run_name}_{i}.png')

def generate_celeba_reconstructions_main():
    model_names = ['vae', 'aef-linear']
    latent_sizes = [256]

    project_name = 'phase21'

    architecture_size = 'big'
    dataset = 'celebahq64'
    img_dim = [3, 64, 64]
    alpha = 0.05
    data_dir = 'celebahq64'

    api = wandb.Api()
    use_gpu = True
    device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")

    _, validation_dataloader, _, _ = get_train_val_dataloaders(dataset, 4, data_dir=data_dir)

    for model_idx, model_name in enumerate(model_names):
        for latent_idx, latent_dims in enumerate(latent_sizes):
            runs = api.runs(path=f"nae/{project_name}", filters={
                "config.latent_dims": latent_dims,
                "config.model": model_name,
                "config.dataset": dataset
            })
            for run_idx, run in enumerate(runs):
                run_id = run.id
                experiment_name = run.name

                model_name = get_field_from_config(run, "model")

                dataset = get_field_from_config(run, "dataset")

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

                for i in range(20):
                    img = plot_reconstructions(model, validation_dataloader, device, img_dim, n_rows=2, n_cols=3, skip_batches=i, padding=0)
                    img.save(f'recsceleba64/{run_name}_{i}.png')

def generate_imagenet_reconstructions_main():
    model_names = ['vae', 'aef-linear']
    latent_sizes = [256]

    project_name = 'phase21'

    architecture_size = 'big'
    dataset = 'imagenet'
    img_dim = [3, 32, 32]
    alpha = 0.05
    data_dir = 'imagenet'

    api = wandb.Api()
    use_gpu = True
    device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")

    _, validation_dataloader, _, _ = get_train_val_dataloaders(dataset, 4, data_dir=data_dir)

    for model_idx, model_name in enumerate(model_names):
        for latent_idx, latent_dims in enumerate(latent_sizes):
            runs = api.runs(path=f"nae/{project_name}", filters={
                "config.latent_dims": latent_dims,
                "config.model": model_name,
                "config.dataset": dataset
            })
            for run_idx, run in enumerate(runs):
                if run.state != 'finished':
                    continue
                if 'continued' not in run.name:
                    continue
                run_id = run.id
                experiment_name = run.name

                model_name = get_field_from_config(run, "model")

                dataset = get_field_from_config(run, "dataset")

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



                for i in range(20):
                    img = plot_reconstructions(model, validation_dataloader, device, img_dim, n_rows=2, n_cols=3, skip_batches=i, padding=0)


                    img.save(f'recs/{run_name}_{i}.png')


def generate_celeba_reconstructions_supp():
    model_names = ['vae']
    latent_sizes = [64, 128, 256]

    project_name = 'phase2'

    architecture_size = 'big'
    img_dim = [3, 32, 32]
    alpha = 0.05

    api = wandb.Api()
    use_gpu = True
    device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")
    k = 0
    for model_idx, model_name in enumerate(model_names):
        for latent_idx, latent_dims in enumerate(latent_sizes):
            runs = api.runs(path="nae/phase2", filters={
                "config.latent_dims": latent_dims,
                "config.model": model_name,
                "config.preprocessing": True
            })
            for run_idx, run in enumerate(runs):
                run_id = run.id
                experiment_name = run.name

                model_name = get_field_from_config(run, "model")

                dataset = get_field_from_config(run, "dataset")

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
                test_loader = get_test_dataloader(dataset, data_dir="celebahq")
                for i in range(2):
                    # fig = plot_reconstructions(model, test_loader, device, img_dim, n_rows=8,
                    #                            skip_batches=k+i)
                    #
                    # plt.savefig(f'recs/{run_name}_{i}.pdf', pad_inches=0, bbox_inches='tight')
                    grid = plot_reconstructions(model, test_loader, device, img_dim, n_rows=8,
                                                skip_batches=k+i)
                    img = torchvision.transforms.ToPILImage()(grid)
                    img.save(f'recspng/{run_name}_{i}.png')


def generate_phase1_reconstructions_and_samples():

    datasets = ['mnist', 'fashionmnist', 'kmnist']
    model_names = ['vae']
    latent_dims = 32
    api = wandb.Api()
    architecture_size = 'small'
    img_dims = [1,28,28]
    alpha = 1e-6

    use_gpu = True
    device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")


    for dataset in datasets:
        for model_name in model_names:
            if model_name == 'vae':
                model_name = 'vae'
                runs = api.runs(path=f"nae/phase1",
                                filters={"config.dataset": dataset,
                                         "config.latent_dims": latent_dims,
                                         "config.model": 'vae',
                                         "config.preprocessing": True
                                         })
            else:
                runs = api.runs(path=f"nae/phase1",
                                filters={"config.dataset": dataset,
                                         "config.latent_dims": latent_dims,
                                         "config.model": model_name,
                                         })

            for run in runs:
                run_id = run.id
                experiment_name = run.name

                try:
                    posterior_flow = get_field_from_config(run, 'posterior_flow')
                    if posterior_flow is None:
                        posterior_flow = 'none'
                    prior_flow = get_field_from_config(run, 'prior_flow')
                    if prior_flow is None:
                        prior_flow = 'none'

                    decoder = get_field_from_config(run, 'decoder')
                    model = get_model(model_name, architecture_size, decoder, latent_dims, img_dims, alpha,
                                      posterior_flow,
                                      prior_flow)
                    run_name = run.name
                    artifact = api.artifact(
                        f'nae/phase1/{run_name}_best:latest')  # run.restore(f'{run_name}_best:latest', run_path=run.path, root='./artifacts')
                    artifact_dir = artifact.download()
                    artifact_dir = artifact_dir + '/' + os.listdir(artifact_dir)[0]
                    model.load_state_dict(torch.load(artifact_dir, map_location=device))
                    model = model.to(device)

                    test_loader = get_test_dataloader(dataset)

                    for i in range(5):
                        img = plot_reconstructions(model, test_loader, device, img_dims, n_rows=8,
                                                   skip_batches=i)

                        img.save(f'mnist_recs/{run_name}_{i}.png')

                        img = plot_samples(model, img_dims, n_rows=8, n_cols=8)

                        img.save(f'mnist_samples/{run_name}_{i}.png')

                except Exception as E:
                    print(E)
                    print(f'Failed to plot samples of {experiment_name}')
                    traceback.print_exc()
                    continue
                break

def generate_celeba64_samples_temperatures():
    model_names = ['aef-linear', 'vae']
    latent_sizes = [64, 256]

    project_name = 'phase21'
    dataset = 'celebahq64'
    architecture_size = 'big'
    img_dim = [3, 64, 64]
    alpha = 0.05

    api = wandb.Api()
    use_gpu = True
    device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")

    temperatures = [0.85]

    for model_idx, model_name in enumerate(model_names):
        for latent_idx, latent_dims in enumerate(latent_sizes):
            runs = api.runs(path=f"nae/{project_name}", filters={
                "config.latent_dims": latent_dims,
                "config.model": model_name,
                "config.dataset": dataset,
            })
            for run_idx, run in enumerate(runs):
                run_id = run.id
                experiment_name = run.name

                model_name = get_field_from_config(run, "model")

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

                for temperature in temperatures:
                    for i in range(10):
                        img = plot_samples(model, img_dim, n_rows=2, n_cols=8, batch_size=16, padding=0,
                                           temperature=temperature)

                        img.save(f'samples/{run_name}_{temperature}_{i}.png')


def denoising_plot_phase1(df):
    datasets = ['mnist', 'kmnist', 'fashionmnist']
    dataset_titles = {'mnist':'MNIST', 'kmnist': 'KMNIST', 'fashionmnist' : 'FashionMNIST'}
    models = ['ae', 'vae', 'aef-linear']
    noise_levels = [0.25, 0.5, 0.75, 1]
    latent_sizes = [2, 32]

    plt.rcParams['axes.axisbelow'] = True

    for dataset in datasets:
        for latent_dims in latent_sizes:
            df_to_use = df.loc[(df.loc[:, 'dataset'] == dataset) & (df.loc[:, 'preprocessing'] == True)
                               & (df.loc[:, 'latent_dims'] == latent_dims)]
            df_to_use = df_to_use.loc[df_to_use.loc[:, 'noise_level'].isin(noise_levels)]
            df_to_use = df_to_use.loc[df_to_use.loc[:, 'model'].isin(models)]
            df_to_use = df_to_use.replace(to_replace={'ae': "AE", 'vae': "VAE", 'aef-linear': 'AEF (linear)'})
            df_to_use = df_to_use.sort_values(by=['model'])

            fig = plt.figure(dpi=300, figsize=(6,3))
            ax = sns.pointplot(x="noise_level", y="ife", hue="model", data=df_to_use, ci=95)

            bottom, top = plt.ylim()  # return the current ylim
            plt.ylim((bottom, top + 0.01))  # set the ylim to bottom, top

            ax.set_facecolor('lavender')
            ax.yaxis.set_major_locator(plt.MaxNLocator(4))
            ax.grid(visible=True, which='major', axis='both', color='w')

            ax.legend(ncol=3, loc='upper center')
            ax.set_xlabel('Noise level')
            ax.set_ylabel('IFE')
            plt.title(dataset_titles[dataset])
            plt.savefig(f'plots/iclr/denoising_{dataset}_latents_{latent_dims}.pdf', bbox_inches='tight')

def denoising_plot_phase2(df):
    models = ['ae', 'vae', 'aef-linear']
    model_titles = ['AE', 'VAE', 'AEF (linear)']
    noise_levels = [0.05, 0.1, 0.2]

    plt.rcParams['axes.axisbelow'] = True

    df_to_use = df
    df_to_use = df_to_use.loc[df_to_use.loc[:, 'noise_level'].isin(noise_levels)]
    df_to_use = df_to_use.replace(to_replace={'ae': "AE", 'vae': "VAE", 'aef-linear': 'AEF (linear)'})
    df_to_use = df_to_use.sort_values(by=['model'])

    fig = plt.figure(dpi=300, figsize=(6,3))
    ax = sns.pointplot(x="noise_level", y="ife", hue="model", data=df_to_use, ci=95)
    bottom, top = plt.ylim()  # return the current ylim
    plt.ylim((bottom, top + 0.01))  # set the ylim to bottom, top

    ax.set_facecolor('lavender')
    ax.yaxis.set_major_locator(plt.MaxNLocator(4))
    ax.grid(visible=True, which='major', axis='both', color='w')

    ax.legend(ncol=3, loc='upper center')
    ax.set_xlabel('Noise level')
    ax.set_ylabel('IFE')
    plt.title('CelebA-HQ (32x32)')
    plt.savefig(f'plots/iclr/denoising_celebahq.pdf', bbox_inches='tight')


def phase1_bpp_fid_plot(df):
    datasets = ['mnist', 'kmnist', 'fashionmnist']
    models_main = ['vae', 'aef-linear']
    models_full = ['vae', 'aef-linear' , 'aef-corner', 'aef-center']
    dataset_titles = {'mnist': 'MNIST', 'kmnist': 'KMNIST', 'fashionmnist': 'FashionMNIST'}

    # todo
    rc = {
        "axes.axisbelow": True,
        }
    plt.rcParams.update(rc)

    for dataset in datasets:

        df_to_use = df[(df.loc[:, 'dataset'] == dataset) & (df.loc[:, 'preprocessing'] == True) & (df.loc[:, 'model'].isin(models_main))]
        df_to_use = df_to_use.replace(to_replace={'vae': "VAE",
                                     'aef-center': 'AEF (center)',
                                     'aef-corner': 'AEF (corner)',
                                     'aef-linear': 'AEF (linear)'})
        df_to_use = df_to_use.sort_values(by='model', axis=0)

        fig = plt.figure()
        ax = sns.pointplot(x="latent_dims", y="test_bpp_adjusted", hue="model", data=df_to_use, ci=95)


        ax.set_facecolor('lavender')
        #ax.yaxis.set_major_locator(plt.MaxNLocator(4))
        ax.grid(visible=True, which='major', axis='both', color='w')

        ax.legend(ncol=2, loc='upper center')
        ax.set_xlabel('Latent dimensions')
        ax.set_ylabel('BPD')
        plt.title(dataset_titles[dataset])
        plt.savefig(f'plots/iclr/bpp_{dataset}_main.pdf', bbox_inches='tight')

        df_to_use = df[(df.loc[:, 'dataset'] == dataset) & (df.loc[:, 'preprocessing'] == True) & (
            df.loc[:, 'model'].isin(models_full))]
        df_to_use = df_to_use.replace(to_replace={'vae': "VAE",
                                                  'aef-center': 'AEF (center)',
                                                  'aef-corner': 'AEF (corner)',
                                                  'aef-linear': 'AEF (linear)'})
        df_to_use = df_to_use.sort_values(by='model', axis=0)

        fig = plt.figure()
        ax = sns.pointplot(x="latent_dims", y="test_bpp_adjusted", hue="model", data=df_to_use, ci=95)
        ax.set_facecolor('lavender')
        # ax.yaxis.set_major_locator(plt.MaxNLocator(4))
        ax.grid(visible=True, which='major', axis='both', color='w')

        ax.legend(ncol=2, loc='upper center')
        ax.set_xlabel('Latent dimensions')
        ax.set_ylabel('BPD')
        plt.title(dataset_titles[dataset])
        plt.savefig(f'plots/iclr/bpp_{dataset}_supp.pdf', bbox_inches='tight')

        fig = plt.figure()
        ax = sns.pointplot(x="latent_dims", y="fid", hue="model", data=df_to_use, ci=95)
        ax.set_facecolor('lavender')
        # ax.yaxis.set_major_locator(plt.MaxNLocator(4))
        ax.grid(visible=True, which='major', axis='both', color='w')

        ax.legend(ncol=2, loc='upper center')
        ax.set_xlabel('Latent dimensions')
        ax.set_ylabel('FID')
        plt.title(dataset_titles[dataset])
        plt.savefig(f'plots/iclr/fid_{dataset}.pdf', bbox_inches='tight')


def phase2_bpp_fid_plot(df):
    datasets = ['celebahq', 'celebahq64']
    models = ['vae', 'aef-linear']
    dataset_titles = {'celebahq' : 'CelebA-HQ (32x32)', 'celebahq64': 'CelebA-HQ (64x64)'}

    for dataset in datasets:

        df_to_use = df[df.loc[:, 'dataset'] == dataset]
        df_to_use = df_to_use.sort_values(by=['model'])
        df_to_use = df_to_use.replace(to_replace={'vae': "VAE",
                                     'aef-linear': 'AEF (linear)'})

        fig = plt.figure(dpi=300, figsize=(6, 4))
        ax = fig.gca()

        ## FID
        sns.pointplot(x="latent_dims", y="fid", hue="model", data=df_to_use, ci=95, rug_kws=dict(rasterized=True, zorder=1), ax=ax)
        bottom, top = plt.ylim()  # return the current ylim
        plt.ylim((bottom, top+10))  # set the ylim to bottom, top

        ax.grid(visible=True, which='major', axis='both', color='w', zorder=0.1)
        ax.set_facecolor('lavender')
        ax.set_axisbelow(True)
        plt.ylim()
        plt.ylabel('FID score')
        plt.xlabel("Nr. of latent dimensions")

        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=handles, labels=labels, loc='upper center', ncol=2)#, bbox_to_anchor=(0.5, -0.1))

        ax.set_title(dataset_titles[dataset])
        plt.savefig(f'plots/iclr/fid_phase21_{dataset}.pdf', bbox_inches='tight')
        plt.clf()
        ##
        ## BPP
        fig = plt.figure(dpi=300, figsize=(6, 4))
        ax = fig.gca()

        sns.pointplot(x="latent_dims", y="test_bpp_adjusted", hue="model", data=df_to_use, ci=95, rug_kws=dict(rasterized=True, zorder=1), ax=ax)
        ax.grid(visible=True, which='major', axis='both', color='w', zorder=0.1)
        ax.set_facecolor('lavender')
        ax.set_axisbelow(True)
        bottom, top = plt.ylim()  # return the current ylim
        plt.ylim((bottom, top + 0.2))  # set the ylim to bottom, top


        plt.ylim()
        plt.ylabel('Bits per pixel')
        plt.xlabel("Nr. of latent dimensions")

        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=handles, labels=labels, loc='upper center', ncol=2)  # , bbox_to_anchor=(0.5, -0.1))

        ax.set_title(dataset_titles[dataset])
        plt.savefig(f'plots/iclr/bpp_phase21_{dataset}.pdf', bbox_inches='tight')
        plt.clf()


def mnist_latent_space_grid():
    model_names = ['aef-linear']
    latent_dims = 2

    project_name = 'phase1'
    dataset = 'mnist'
    architecture_size = 'small'
    decoder = 'independent'
    img_dim = [1, 28, 28]
    alpha = 1e-6

    api = wandb.Api()
    use_gpu = True
    device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")

    test_loader = get_test_dataloader(dataset)

    z_vals = get_z_values()

    for model_idx, model_name in enumerate(model_names):
        runs = api.runs(path=f"nae/{project_name}", filters={
            "config.latent_dims": latent_dims,
            "config.model": model_name,
            "config.dataset": dataset,
        })
        for run_idx, run in enumerate(runs):
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

            fig = plot_latent_space_2d(model, test_loader, device, max_val=3.5, colorbar=False)
            plt.savefig(f'plots/iclr/latent_spaces_grids/latent_{run_name}.pdf', bbox_inches='tight')
            plt.clf()

            z_vals = z_vals.to(device)

            with torch.no_grad():
                output = model.decode(z_vals)
                if isinstance(output, tuple):
                    output = output[0]
                output = output.detach().cpu()
            img = plot_image_grid(output, cols=20, padding=0)
            img.save(f'plots/iclr/latent_spaces_grids/grid_{run_name}.png')



def generate_ablation_recs_and_samples():
    project_name = 'ablation-celeba-big'

    dataset = 'celebahq'
    data_dir = 'data/celebahq/celebahq'
    architecture_size = 'big'
    img_dim = [3, 32, 32]
    alpha = 0.05

    api = wandb.Api()
    use_gpu = True
    device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")
    k = 0
    test_loader = get_test_dataloader(dataset, data_dir=data_dir, batch_size=8)

    runs = api.runs(path=f"nae/{project_name}")
    for run_idx, run in enumerate(runs):
        model_name = get_field_from_config(run, "model")

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

        for i in range(10):
            img = plot_reconstructions(model, test_loader, device, img_dim, n_rows=2, padding=0, n_cols=6,
                                       skip_batches=i)
            img.save(f'recs/rec_{run_name}_{i}.png')
            img = plot_samples(model, img_dim, n_rows=2, n_cols=6, batch_size=12, temperature=0.85, padding=0)
            img.save(f'recs/sam_{run_name}_{i}.png')

def generate_phase1_recs_and_samples():
    project_name = 'phase1'

    datasets = ['mnist', 'fashionmnist', 'kmnist']

    architecture_size = 'small'
    img_dim = [1, 28, 28]
    alpha = 1e-6
    latent_dims = 32

    api = wandb.Api()
    use_gpu = True
    device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")

    model_names = ['aef-linear', 'vae']

    for dataset in datasets:
        test_loader = get_test_dataloader(dataset, batch_size=8)
        for model_name in model_names:
            runs = api.runs(path=f"nae/{project_name}", filters={'config.dataset': dataset,
                                                                 'config.model': model_name,
                                                                 'config.latent_dims': latent_dims})
            for run_idx, run in enumerate(runs):
                model_name = get_field_from_config(run, "model")

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

                for i in range(10):
                    img = plot_reconstructions(model, test_loader, device, img_dim, n_rows=2, padding=0, n_cols=6,
                                               skip_batches=i)
                    img.save(f'phase1/rec_{run_name}_{i}.png')
                    img = plot_samples(model, img_dim, n_rows=2, n_cols=6, batch_size=12, temperature=0.85, padding=0)
                    img.save(f'phase1/sam_{run_name}_{i}.png')
                break

def generate_phase1ablation_recs_and_samples():
    project_name = 'ablation-mnist-small'

    dataset = 'mnist'
    architecture_size = 'small'
    img_dim = [1, 28, 28]
    alpha = 1e-6

    api = wandb.Api()
    use_gpu = True
    device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")
    test_loader = get_test_dataloader(dataset, batch_size=8)

    runs = api.runs(path=f"nae/{project_name}")
    for run_idx, run in enumerate(runs):
        model_name = get_field_from_config(run, "model")

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

        for i in range(10):
            img = plot_reconstructions(model, test_loader, device, img_dim, n_rows=2, padding=0, n_cols=6,
                                       skip_batches=i)
            img.save(f'phase1_ablations/rec_{run_name}_{i}.png')
            img = plot_samples(model, img_dim, n_rows=2, n_cols=6, batch_size=12, temperature=0.85, padding=0)
            img.save(f'phase1_ablations/sam_{run_name}_{i}.png')

if __name__ == "__main__":
    plt.rcParams.update(bundles.iclr2023())
    sys.exit(0)
