import torch
import torch.nn as nn
from torch.distributions import Normal
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, hidden_channels: int, latent_dim: int, input_dim):
        """
        Simple encoder module

        It predicts the `mean` and `log(variance)` parameters.

        The choice to use the `log(variance)` is for stability reasons:
        https://stats.stackexchange.com/a/353222/284141
        """
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=input_dim[0],
                               out_channels=hidden_channels,
                               kernel_size=3,
                               stride=2,
                               padding=1)

        self.conv2 = nn.Conv2d(in_channels=hidden_channels,
                               out_channels=hidden_channels * 2,
                               kernel_size=3,
                               stride=2,
                               padding=1)

        self.fc_mu = nn.Linear(in_features=hidden_channels * 2 * input_dim[1]//4 * input_dim[2]//4,
                               out_features=latent_dim)
        self.fc_logvar = nn.Linear(in_features=hidden_channels * 2 * input_dim[1]//4 * input_dim[2]//4,
                                   out_features=latent_dim)

        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor):
        """
        :param x: batch of images with shape [batch, channels, w, h]
        :returns: the predicted mean and log(variance)
        """
        batch_size = x.shape[0]
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))

        x = x.view(x.shape[0], -1)

        x_mu = self.fc_mu(x)
        x_mu = x_mu
        x_logvar = self.fc_logvar(x)
        x_logvar = x_logvar

        return x_mu.view(batch_size, -1), F.softplus(x_logvar).view(batch_size,
                                                                    -1)

class LatentDependentDecoder(nn.Module):
    def __init__(self, hidden_channels, latent_dim, output_shape):
        """
        Simple decoder module
        """
        super().__init__()

        self.output_shape = output_shape
        self.hidden_channels = hidden_channels

        # out features will work for images of size 28x28. 32x32 and 64x64
        # would crash for sizes that are not divisible by 4
        self.fc = nn.Linear(in_features=latent_dim,
                            out_features=hidden_channels * 2 * output_shape[1]//4 * output_shape[2]//4)

        self.conv2 = nn.ConvTranspose2d(in_channels=hidden_channels * 2,
                                        out_channels=hidden_channels,
                                        kernel_size=4,
                                        stride=2,
                                        padding=1)
        self.conv1 = nn.ConvTranspose2d(in_channels=hidden_channels,
                                        out_channels=output_shape[0] * 2,
                                        kernel_size=4,
                                        stride=2,
                                        padding=1)

        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor):
        """
        :param x: a sample from the distribution governed by the mean and log(var)
        :returns: a reconstructed image with size [batch, 1, w, h]
        """
        x = self.fc(x)
        x = x.view(x.size(0), self.hidden_channels * 2, self.output_shape[1]//4, self.output_shape[2]//4)
        x = self.activation(self.conv2(x))
        x = self.conv1(x)
        return x[:,:self.output_shape[0]], F.softplus(x[:,self.output_shape[0]:])

class IndependentVarianceDecoder(nn.Module):
    def __init__(self, hidden_channels, latent_dim, output_shape):
        """
        Simple decoder module
        """
        super().__init__()

        self.output_shape = output_shape
        self.hidden_channels = hidden_channels

        # out features will work for images of size 28x28. 32x32 and 64x64
        # would crash for sizes that are not divisible by 4
        self.fc = nn.Linear(in_features=latent_dim,
                            out_features=hidden_channels * 2 * output_shape[1]//4 * output_shape[2]//4)

        self.conv2 = nn.ConvTranspose2d(in_channels=hidden_channels*2,
                                        out_channels=hidden_channels,
                                        kernel_size=4,
                                        stride=2,
                                        padding=1)
        self.conv1 = nn.ConvTranspose2d(in_channels=hidden_channels,
                                        out_channels=output_shape[0],
                                        kernel_size=4,
                                        stride=2,
                                        padding=1)

        self.activation = nn.ReLU()
        self.pre_sigma = nn.Parameter(torch.zeros(output_shape))

    def forward(self, x: torch.Tensor):
        """
        :param x: a sample from the distribution governed by the mean and log(var)
        :returns: a reconstructed image with size [batch, 1, w, h]
        """
        x = self.fc(x)
        x = x.view(x.size(0), self.hidden_channels * 2, self.output_shape[1]//4, self.output_shape[2]//4)
        x = self.activation(self.conv2(x))
        x = self.conv1(x)
        return x, F.softplus(self.pre_sigma)




