import torch
import wandb
from util import load_best_model
import matplotlib.pyplot as plt

run = wandb.init()
project_name = 'test-project'
model_name = 'vae-iaf'#'nae'
experiment_name = 'model_best'#'nae_fashionmnist_latent_size(16)_0_indvar_center'
latent_dims = 16
image_dim = [1,28,28]
alpha = 1e-6
use_center_pixels = True
device = 'cpu'
version = 'v58' # v75 --> VAE, v78 -->IWAE v58 --> vae-iaf
decoder = 'independent'

model = load_best_model(run, project_name, model_name, experiment_name, device, decoder, latent_dims, image_dim, alpha, use_center_pixels, version=version)

for temperature in [0.1, 0.5, 2, 5]:
    samples = model.sample(16, temperature=temperature)
    samples = samples.cpu().detach().numpy()
    _, axs = plt.subplots(4, 4, )
    axs = axs.flatten()
    for img, ax in zip(samples, axs):
        ax.axis('off')
        ax.imshow(img.reshape(28, 28), cmap='gray')

    plt.suptitle(f'temperature: {temperature}')

    plt.show()
