import torch

import wandb
import pandas as pd

from util import load_best_model, vae_log_prob
from datasets import get_test_dataloader


def extract_data_from_runs():
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
    use_center_pixels = []

    runs = api.runs(path="nae/phase1")
    for run in runs:
        dataset.append(get_field_from_config(run, "dataset"))
        decoder.append(get_field_from_config(run, "decoder"))
        latent_dims.append(get_field_from_config(run, "latent_dim", type="int"))
        architecture_size.append(get_field_from_config(run, "architecture_size"))
        prior_flow.append(get_field_from_config(run, "prior_flow"))
        posterior_flow.append(get_field_from_config(run, "posterior_flow"))
        test_loss.append(get_field_from_summary(run, "test_loss", type="float"))
        test_bpp.append(get_field_from_summary(run, "test_bpp", type="float"))
        test_bpp_adjusted.append(get_field_from_summary(run, "test_bpp_adjusted", type="float"))
        train_loss.append(get_field_from_summary(run, "train_loss", type="float"))
        val_loss.append(get_field_from_summary(run, "val_loss", type="float"))

        # Backwards compatibility: before we used 'nae' for both 'nae-center' and 'nae-corner'.
        model_name = get_field_from_config(run, "model")
        if model_name[-1] == 'nae':
            if run.name.split('_')[-1] == 'corner':
                model_name = 'nae-corner'
                use_center_pixels.append(False)
            elif run.name.split('_')[-1] == 'center':
                model_name = 'nae-center'
                use_center_pixels.append(True)
            else:
                print('Encountered something weird.')
        else:
            if 'nae' in model_name:
                if model_name == 'nae-center':
                    use_center_pixels.append(True)
                else:
                    use_center_pixels.append(False)
            else:
                use_center_pixels.append(None)
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
                }
    df = pd.DataFrame(col_dict)

    return df

def check_for_outliers(df, model_name, latent_dims):
    rows = df[df['model'] == model_name & df['latent_dim'] == latent_dims]



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
    df = extract_data_from_runs()
    print(df)

