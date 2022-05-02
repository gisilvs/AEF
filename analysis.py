import torch

import wandb
import pandas as pd

from util import load_best_model, vae_log_prob
from datasets import get_test_dataloader


def extract_data_from_runs():
    api = wandb.Api(timeout=19)

    model_name = []
    run_nr = []
    dataset = []
    decoder = []
    latent_dims = []
    test_loss = []
    test_bpp = []
    test_bpp_adjusted = []
    train_loss = []
    val_loss = []
    use_center_pixels = []

    runs = api.runs(path="nae/phase1")
    for run in runs:
        model_name.append(get_field_from_config(run, "model"))
        dataset.append(get_field_from_config(run, "dataset"))
        decoder.append(get_field_from_config(run, "decoder"))
        latent_dims.append(get_field_from_config(run, "latent_dims"))
        test_loss.append(get_field_from_summary(run, "test_loss"))
        test_bpp.append(get_field_from_summary(run, "test_bpp"))
        test_bpp_adjusted.append(get_field_from_summary(run, "test_bpp_adjusted"))
        train_loss.append(get_field_from_summary(run, "train_loss"))
        val_loss.append(get_field_from_summary(run, "val_loss"))
        if model_name[-1] == 'nae':
            if run.name.split('_')[-1] == 'corner':
                use_center_pixels.append(True)
            else:
                use_center_pixels.append(False)
        else:
            use_center_pixels.append(None)

        run_nr_idx = run.name.find('run_')
        if run_nr_idx != -1:
            run_nr.append(int(run.name[run_nr_idx+4]))
        else:
            run_nr.append(None)

    col_dict = {'model' : model_name,
            'dataset': dataset,
            'latent_dims': latent_dims,
            'decoder': decoder,
            'test_loss': test_loss,
            'test_bpp': test_bpp,
            'test_bpp_adjusted': test_bpp_adjusted,
            'train_loss': train_loss,
            'val_loss': val_loss
            }
    df = pd.DataFrame(col_dict)

    return df

def get_field_from_config(run: wandb.run, field: str):
    if field not in run.config.keys():
        return None
    return run.config[field]

def get_field_from_summary(run: wandb.run, field: str):
    if field not in run.summary.keys():
        return None
    return run.summary[field]


def approximate_log_likelihood():
    run = wandb.init()
    project_name = 'phase1'
    image_dim = [1, 28, 28]

    alpha = 1e-6
    use_center_pixels = False
    use_gpu = False
    device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")
    model_name = 'vae'
    dataset = 'mnist'
    latent_dims = 2
    experiment_name = f'{model_name}_{dataset}_run_2_latent_size_{latent_dims}_decoder_fixed'
    decoder = 'fixed'
    model = load_best_model(run, project_name, model_name, experiment_name, device, latent_dims, image_dim,
                            alpha, decoder, use_center_pixels, version='best:latest')
    test_loader = get_test_dataloader(dataset)
    batch, _ = next(iter(test_loader))
    log_prob = vae_log_prob(model, batch, 128)
    print(log_prob)


if __name__ == '__main__':
    approximate_log_likelihood()
