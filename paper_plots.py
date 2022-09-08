import os
import sys
import traceback

import pandas as pd
import torch
import torchvision

import wandb
from matplotlib import pyplot as plt
import seaborn as sns

from analysis import get_field_from_config, extract_data_from_runs
from datasets import get_test_dataloader
from models import model_database
from models.model_database import get_model
from visualize import plot_reconstructions, plot_noisy_reconstructions, plot_samples


def generate_celeba_samples_main():
    fig, axs = plt.subplots(2, 3, figsize=(6.4, 4.8), dpi=300)

    model_names = ['vae']
    latent_sizes = [64, 128, 256]

    project_name = 'phase2'

    architecture_size = 'big'
    img_dim = [3, 32, 32]
    alpha = 0.05

    api = wandb.Api()
    use_gpu = True
    device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")

    params = {
        "text.usetex": True,
        "font.family": "Times New Roman",
        'axes.titlesize': 'xx-large',
    }
    plt.rcParams.update(params)
    for i in range(10):
        for model_idx, model_name in enumerate(model_names):
            for latent_idx, latent_dims in enumerate(latent_sizes):
                if model_name == 'vae':

                    runs = api.runs(path="nae/phase2", filters={
                        "config.latent_dims": latent_dims,
                        "config.model": model_name,
                        "config.preprocessing": True
                    })
                else:
                    runs = api.runs(path="nae/phase2", filters={
                        "config.latent_dims": latent_dims,
                        "config.model": model_name,
                    })
                for run in runs:
                    run_id = run.id
                    experiment_name = run.name

                    model_name = get_field_from_config(run, "model")
                    if model_name == 'nae-external':
                        model_name = 'aef-linear'

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

                    samples = model.sample(4).detach().cpu()
                    grid_img = torchvision.utils.make_grid(samples, padding=0, pad_value=0., nrow=2)
                    axs[model_idx, latent_idx].imshow(grid_img.permute(1, 2, 0))
                    axs[model_idx, latent_idx].axis("off")
                    if model_idx == 0:
                        axs[model_idx, latent_idx].set_title(f'{latent_dims}')

                    if latent_idx == 0:
                        lbl = 'AEF' if model_idx == 0 else 'VAE'
                        axs[model_idx, latent_idx].set_xlabel(lbl)
        fig.tight_layout()
        plt.savefig(f'plots/celeba_samples_{i}.pdf', dpi=300, bbox_inches='tight')
        plt.show()


def generate_denoising_reconstructions_main():
    dataset = 'celebahq'
    models = ['ae', 'vae', 'aef-linear']

    latent_dims = 128
    api = wandb.Api()
    img_dim = [3, 32, 32]
    alpha = 0.05
    project_name = 'denoising-experiments-2'
    noise_level = 0.1
    architecture_size = 'big'

    use_gpu = True
    device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")

    noise_distribution = torch.distributions.normal.Normal(torch.zeros([60, *img_dim]),
                                                           noise_level * torch.ones([60, *img_dim]))
    test_loader = get_test_dataloader(dataset, data_dir='celebahq')
    batch_iter = iter(test_loader)

    for i in range(10):
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

                    fig = plot_noisy_reconstructions(model, image_batch, device, noise_distribution,
                                                     img_dim, n_rows=6, n_cols=6)
                    plt.savefig(f'denoising_main/denoising_{run_name}_{i}.pdf', bbox_inches='tight', pad_inches=0)
                except Exception as E:
                    print(E)
                    print(f'Failed to plot latent space of {experiment_name}')
                    traceback.print_exc()
                    continue
            plt.close('all')

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




def generate_celeba_samples_supp():
    model_names = ['nae-external']
    latent_sizes = [64, 128, 256]

    project_name = 'phase2'

    architecture_size = 'big'
    img_dim = [3, 32, 32]
    alpha = 0.05

    api = wandb.Api()
    use_gpu = True
    device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")

    for model_idx, model_name in enumerate(model_names):
        for latent_idx, latent_dims in enumerate(latent_sizes):
            runs = api.runs(path="nae/phase2", filters={
                "config.latent_dims": latent_dims,
                "config.model": model_name,
                #"config.preprocessing": True,
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
                for temp in [0.4, 0.6, 0.8, 1.0]: #[0.4, 0.6, 0.8, 1.0]:
                    for i in range(3):
                        grid = plot_samples(model, img_dim, n_rows=8, n_cols=8, batch_size=64, temperature=temp)
                        img = torchvision.transforms.ToPILImage()(grid)
                        img.save(f'samples/{run_name}_{temp}_{i}.png')

                plt.close("all")

def generate_celeba_samples_main():
    model_names = ['vae', 'aef-linear']
    latent_sizes = [128, 256, 512]

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

                for i in range(50):
                    img = plot_samples(model, img_dim, n_rows=2, n_cols=2, batch_size=64, padding=0, temperature=0.85)

                    img.save(f'celeba_samples2/{run_name}_{i}.png')




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
    latent_sizes = [128, 256]

    project_name = 'phase21'
    dataset = 'celebahq64'
    architecture_size = 'big'
    img_dim = [3, 64, 64]
    alpha = 0.05

    api = wandb.Api()
    use_gpu = True
    device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")

    temperatures = [1.0, 0.95, 0.9, 0.85, 0.8, 0.75]

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
                    for i in range(3):
                        img = plot_samples(model, img_dim, n_rows=4, n_cols=4, batch_size=64, padding=1,
                                           temperature=temperature)

                        img.save(f'samples/{run_name}_{temperature}_{i}.png')


def denoising_plot(df):
    datasets = ['mnist', 'kmnist', 'fashionmnist']
    dataset_titles = {'mnist':'MNIST', 'kmnist': 'KMNIST', 'fashionmnist' : 'FashionMNIST'}
    models = ['ae', 'nae-external', 'vae-iaf-maf']
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
    row_indexer = (df.loc[:, 'model'] == 'vae') \
                  & (df.loc[:, 'posterior_flow'] == 'iaf') \
                  & (df.loc[:, 'prior_flow'] == 'maf')
    df.loc[row_indexer, 'model'] = 'vae-iaf-maf'

    for dataset in datasets:
        df_to_use = df.loc[df.loc[:, 'dataset'] == dataset]
        df_to_use = df_to_use.loc[df_to_use.loc[:, 'noise_level'].isin(noise_levels)]
        df_to_use = df_to_use.loc[df_to_use.loc[:, 'model'].isin(models)]
        df_to_use = df_to_use.replace(to_replace={'ae': "AE", 'vae-iaf-maf': "VAE-IAF-MAF", 'nae-external': 'AEF'})
        df_to_use = df_to_use.sort_values(by=['model'])

        fig = plt.figure()
        ax = sns.pointplot(x="noise_level", y="test_rce_with_noise", hue="model", data=df_to_use, ci=95)
        ax.set_facecolor('lavender')
        ax.yaxis.set_major_locator(plt.MaxNLocator(4))
        ax.grid(visible=True, which='major', axis='both', color='w')

        ax.legend(ncol=3)
        ax.set_xlabel('Noise level')
        ax.set_ylabel('Avg. reconstruction error')
        plt.title(dataset_titles[dataset])
        plt.savefig(f'plots/denoising_{dataset}.pdf')

def phase1_bpp_plot(df, broken_axis=True):
    datasets = ['mnist', 'kmnist', 'fashionmnist']
    models = ['ae', 'vae-iaf', 'nae-external']
    dataset_titles = {'mnist': 'MNIST', 'kmnist': 'KMNIST', 'fashionmnist': 'FashionMNIST'}


    vae_models = ['vae', 'iwae', 'vae-iaf']
    nae_models = ['nae-external', 'nae-corner', 'nae-center']

    # Replace names



    df_fixed = df.loc[(df.loc[:,'latent_dims'] <= 32) & (df.loc[:, 'model'] != 'maf'), :]
    row_indexer = (df_fixed.loc[:, 'model'] == 'vae') \
                  & (df_fixed.loc[:, 'posterior_flow'] == 'iaf') \
                  & (df_fixed.loc[:, 'prior_flow'] == 'maf')
    df_fixed.loc[row_indexer, 'model'] = 'vae-iaf-maf'


    # todo
    # rc = {
    #     "text.usetex": True,
    #     "font.family": "Times New Roman",
    #     }
    # plt.rcParams.update(rc)

    for dataset in datasets:

        maf_mean = df.loc[(df.loc[:, 'model'] == 'maf') & (df.loc[:, 'dataset'] == dataset), 'test_bpp_adjusted'].mean()

        df_to_use = df_fixed[df.loc[:, 'dataset'] == dataset]
        #df_to_use = df_to_use[df.loc[:, 'model'].isin(['vae', 'vae-iaf', 'iwae'])]
        df_to_use = df_to_use.replace(to_replace={'iwae': "vae-iwae"}) #hack to get right ordering
        df_to_use = df_to_use.sort_values(by=['model'])
        df_to_use = df_to_use.replace(to_replace={'vae-iwae': "iwae"})
        df_to_use = df_to_use.replace(to_replace={'vae': "VAE",
                                     'iwae': "IWAE",
                                     'vae-iaf': "VAE-IAF",
                                     'vae-iaf-maf': "VAE-IAF-MAF",
                                     'nae-center': 'IAE (center)',
                                     'nae-corner': 'IAE (corner)',
                                     'nae-external': 'IAE (linear)'})

        if not broken_axis:
            ax = sns.pointplot(x="latent_dims", y="test_bpp_adjusted", hue="model", data=df_to_use, ci=95)
        else:

            top_scores = df_to_use.loc[df.loc[:, 'model'].isin(vae_models), 'test_bpp_adjusted']
            bottom_scores = df_to_use.loc[df.loc[:, 'model'].isin(nae_models), 'test_bpp_adjusted']
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

            maf_line_handle = ax_bottom.axhline(y=maf_mean, c='k', linestyle='--', label='MAF')

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

            handles, labels = ax_top.get_legend_handles_labels()
            handles.append(maf_line_handle)
            labels.append("MAF")
            ax_top.legend(handles=handles, labels=labels, loc='upper center', ncol=3)#, bbox_to_anchor=(0.5, -0.1))
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

def phase1_fid_plot(df):
    datasets = ['mnist', 'kmnist', 'fashionmnist']
    models = ['ae', 'vae-iaf', 'nae-external']
    dataset_titles = {'mnist': 'MNIST', 'kmnist': 'KMNIST', 'fashionmnist': 'FashionMNIST'}


    vae_models = ['vae', 'iwae', 'vae-iaf']
    nae_models = ['nae-external', 'nae-corner', 'nae-center']

    # Replace names

    df_fixed = df.loc[(df.loc[:,'latent_dims'] <= 32) & (df.loc[:, 'model'] != 'maf'), :]
    row_indexer = (df_fixed.loc[:, 'model'] == 'vae') \
                  & (df_fixed.loc[:, 'posterior_flow'] == 'iaf') \
                  & (df_fixed.loc[:, 'prior_flow'] == 'maf')
    df_fixed.loc[row_indexer, 'model'] = 'vae-iaf-maf'

    # todo
    # rc = {
    #     "text.usetex": True,
    #     "font.family": "Times New Roman",
    #     }
    # plt.rcParams.update(rc)

    for dataset in datasets:

        maf_mean = df.loc[(df.loc[:, 'model'] == 'maf') & (df.loc[:, 'dataset'] == dataset), 'fid'].mean()

        df_to_use = df_fixed[df.loc[:, 'dataset'] == dataset]
        #df_to_use = df_to_use[df.loc[:, 'model'].isin(['vae', 'vae-iaf', 'iwae'])]
        df_to_use = df_to_use.replace(to_replace={'iwae': "vae-iwae"}) #hack to get right ordering
        df_to_use = df_to_use.sort_values(by=['model'])
        df_to_use = df_to_use.replace(to_replace={'vae-iwae': "iwae"})
        df_to_use = df_to_use.replace(to_replace={'vae': "VAE",
                                     'iwae': "IWAE",
                                     'vae-iaf': "VAE-IAF",
                                     'vae-iaf-maf': "VAE-IAF-MAF",
                                     'nae-center': 'IAE (center)',
                                     'nae-corner': 'IAE (corner)',
                                     'nae-external': 'IAE (linear)'})



        fig = plt.figure(dpi=300, figsize=(6,6))
        ax = fig.gca()
        sns.pointplot(x="latent_dims", y="fid", hue="model", data=df_to_use, ci=95)

        maf_line_handle = ax.axhline(y=maf_mean, c='k', linestyle='--', label='MAF')

        bottom, top = plt.ylim()  # return the current ylim
        plt.ylim((bottom, top+20))  # set the ylim to bottom, top

        ax.grid(visible=True, which='major', axis='both', color='w')

        #sns.set_theme()
        ax.set_facecolor('lavender')

        # p.yaxis.set_major_locator(plt.MaxNLocator(4))
        # ax_bottom.yaxis.set_major_locator(plt.MaxNLocator(4))
        plt.ylim()
        plt.ylabel('FID score')
        plt.xlabel("Nr. of latent dimensions")

        handles, labels = ax.get_legend_handles_labels()
        # handles.append(maf_line_handle)
        # labels.append("MAF")
        ax.legend(handles=handles, labels=labels, loc='upper center', ncol=3)#, bbox_to_anchor=(0.5, -0.1))

        #fig.subplots_adjust(bottom=0.2)
        # handles = ax_top.legend_.data.values()
        # labels = ax_top.legend_.data.keys()
        #
        # ax_bottom.legend(handles=handles, labels=labels, loc='lower center', ncol=6)

        # Shrink current axis's height by 10% on the bottom
        # box = ax.get_position()
        # ax.set_position([box.x0, box.y0 + box.height * 0.1,
        #                  box.width, box.height * 0.9])

        ax.set_title(dataset_titles[dataset])
        plt.savefig(f'plots/fid_phase1_{dataset}.png')

def phase2_bpp_fid_plot(df):
    datasets = ['celebahq', 'celebahq64']
    models = ['vae', 'aef-linear']
    dataset_titles = {'celebahq' : 'CelebA-HQ (32x32)', 'celebahq64': 'CelebA-HQ (64x64)'}

    # Replace names

    # todo
    # rc = {
    #     "text.usetex": True,
    #     "font.family": "Times New Roman",
    #     }
    # plt.rcParams.update(rc)

    for dataset in datasets:

        df_to_use = df[df.loc[:, 'dataset'] == dataset]
        df_to_use = df_to_use.sort_values(by=['model'])
        df_to_use = df_to_use.replace(to_replace={'vae': "VAE",
                                     'aef-linear': 'AEF (linear)'})

        fig = plt.figure(dpi=300, figsize=(6, 6))
        ax = fig.gca()

        ## FID
        sns.pointplot(x="latent_dims", y="fid", hue="model", data=df_to_use, ci=95, rug_kws=dict(rasterized=True, zorder=1), ax=ax)
        bottom, top = plt.ylim()  # return the current ylim
        #plt.ylim((bottom, top+20))  # set the ylim to bottom, top

        ax.grid(visible=True, which='major', axis='both', color='w', zorder=0.1)
        ax.set_facecolor('lavender')
        ax.set_axisbelow(True)
        plt.ylim()
        plt.ylabel('FID score')
        plt.xlabel("Nr. of latent dimensions")

        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=handles, labels=labels, loc='upper center', ncol=2)#, bbox_to_anchor=(0.5, -0.1))

        ax.set_title(dataset_titles[dataset])
        plt.savefig(f'plots/fid_phase21_{dataset}.pdf')
        plt.clf()
        ##
        ## BPP
        fig = plt.figure(dpi=300, figsize=(6, 6))
        ax = fig.gca()

        sns.pointplot(x="latent_dims", y="test_bpp_adjusted", hue="model", data=df_to_use, ci=95, rug_kws=dict(rasterized=True, zorder=1), ax=ax)
        ax.grid(visible=True, which='major', axis='both', color='w', zorder=0.1)
        ax.set_facecolor('lavender')
        ax.set_axisbelow(True)
        bottom, top = plt.ylim()  # return the current ylim
        #plt.ylim((bottom, top + 0.2))  # set the ylim to bottom, top


        plt.ylim()
        plt.ylabel('Bits per pixel')
        plt.xlabel("Nr. of latent dimensions")

        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=handles, labels=labels, loc='upper center', ncol=2)  # , bbox_to_anchor=(0.5, -0.1))

        ax.set_title(dataset_titles[dataset])
        plt.savefig(f'plots/bpp_phase21_{dataset}.pdf')
        plt.clf()


if __name__ == "__main__":
    #generate_celeba_samples_main()
    #generate_celeba64_samples_temperatures()
    #generate_celeba_samples_supp()
    #generate_denoising_reconstructions_main()
    # df = extract_data_from_runs('phase21')
    # df.to_pickle('phase21.pkl')
    df = pd.read_pickle('phase21.pkl')

    rc = {
        "text.usetex": True,
        "font.family": "Times New Roman",
    }
    plt.rcParams.update(rc)
    phase2_bpp_fid_plot(df)
    sys.exit(0)
    # generate_denoising_reconstructions_main()
