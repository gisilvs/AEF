import torch
import matplotlib.pyplot as plt
import normflow as nf

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

from flows.realnvp import RealNVP
from models.autoencoder import Encoder, Decoder
from models.normalizing_autoencoder import NormalizingAutoEncoder

from tqdm import tqdm

from util import make_averager, refresh_bar, plot_loss, dequantize

if __name__ == "__main__":
    # 2-d latent space, parameter count in same order of magnitude
    # as in the original VAE paper (VAE paper has about 3x as many)
    latent_dims = 4
    num_epochs = 2
    batch_size = 128
    capacity = 64
    learning_rate = 1e-3
    variational_beta = 1
    alpha = 1e-6
    use_gpu = True

    device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")

    do_dequantize = True

    img_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_dataset = MNIST(root='./data/MNIST', download=True, train=True, transform=img_transform)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = MNIST(root='./data/MNIST', download=True, train=False, transform=img_transform)
    test_dataloader = DataLoader(test_dataset, batch_size=max(10000, batch_size), shuffle=True)

    core_flow = RealNVP(input_dim=4, num_flows=6, hidden_units=256)
    encoder = Encoder(64,4)
    decoder = Decoder(64, 4, [1,28,28])
    mask = torch.zeros(28,28)
    mask[13:15, 13:15] = 1
    mask = mask.to(device)
    preprocessing_layers = [nf.transforms.Logit(alpha), nf.flows.ActNorm([1,28,28])]
    nae = NormalizingAutoEncoder(core_flow, encoder, decoder, mask, preprocessing_layers)
    optimizer = torch.optim.Adam(params=nae.parameters(), lr=1e-3)#, weight_decay=1e-5)
    nae = nae.to(device)

    train_loss_avg = []

    print('Training ...')

    tqdm_bar = tqdm(range(1, num_epochs + 1), desc="epoch [loss: ...]")
    for epoch in tqdm_bar:
        train_loss_averager = make_averager()

    batch_bar = tqdm(train_dataloader, leave=False, desc='batch',
                     total=len(train_dataloader))
    for image_batch, _ in batch_bar:
        if do_dequantize:
            image_batch = dequantize(image_batch)
        image_batch = image_batch.to(device)

        loss = torch.mean(nae.neg_log_likelihood(image_batch))
        # backpropagation
        optimizer.zero_grad()
        loss.backward()

        # one step of the optmizer
        optimizer.step()

        refresh_bar(batch_bar,
                    f"train batch [loss: {train_loss_averager(loss.item()):.3f}]")

    refresh_bar(tqdm_bar, f"epoch [loss: {train_loss_averager(None):.3f}]")

    train_loss_avg.append(train_loss_averager(None))
    plot_loss(train_loss_avg)
    plt.show()
    samples = nae.sample(16)
    samples = samples.cpu().detach().numpy()
    _, axs = plt.subplots(4, 4, )
    axs = axs.flatten()
    for img, ax in zip(samples, axs):
      ax.axis('off')
      ax.imshow(img.reshape(28,28), cmap='gray')

    plt.show()