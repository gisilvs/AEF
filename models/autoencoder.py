from typing import Tuple, List

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F


class Coder(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError


class GaussianCoder(Coder):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        '''
        A GaussianCoder returns a mu and sigma with the same shape as x.
        :param x:
        :return: mu, sigma
        '''
        raise NotImplementedError


class ConvolutionalEncoder(GaussianCoder):
    def __init__(self, hidden_channels: int, latent_dim: int, input_dim: List):
        '''
        Default convolutional encoder class.
        :param hidden_channels:
        :param latent_dim:
        :param input_dim:
        '''
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

        self.fc_mu = nn.Linear(in_features=hidden_channels * 2 * input_dim[1] // 4 * input_dim[2] // 4,
                               out_features=latent_dim)
        self.fc_sigma = nn.Linear(in_features=hidden_channels * 2 * input_dim[1] // 4 * input_dim[2] // 4,
                                    out_features=latent_dim)

        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor):
        """
        :param x: batch of images with shape [batch, channels, w, h]
        :returns: mu(x), softplus(sigma(x))
        """
        batch_size = x.shape[0]
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))

        x = x.view(batch_size, -1)

        z_mu = self.fc_mu(x)
        z_mu = z_mu.view(batch_size, -1)
        z_sigma = self.fc_sigma(x)
        z_sigma = F.softplus(z_sigma).view(batch_size, -1)

        return z_mu, z_sigma


class LatentDependentDecoder(GaussianCoder):
    def __init__(self, hidden_channels: int, latent_dim: int, output_shape: List):
        """
        Convolutional decoder where sigma is not dependent on z (but is learned).
        """
        super().__init__()

        self.output_shape = output_shape
        self.hidden_channels = hidden_channels

        # out features will work for images of size 28x28. 32x32 and 64x64
        # would crash for sizes that are not divisible by 4
        self.fc = nn.Linear(in_features=latent_dim,
                            out_features=hidden_channels * 2 * output_shape[1] // 4 * output_shape[2] // 4)

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

    def forward(self, z: torch.Tensor):
        """
        :param z:
        :returns: mu(z), sigma(z)
        """
        x = self.fc(z)
        x = x.view(x.size(0), self.hidden_channels * 2, self.output_shape[1] // 4, self.output_shape[2] // 4)
        x = self.activation(self.conv2(x))
        x = self.conv1(x)
        x_mu = x[:, :self.output_shape[0]]
        x_sigma = F.softplus(x[:, self.output_shape[0]:])
        return x_mu, x_sigma


class IndependentVarianceDecoder(GaussianCoder):
    def __init__(self, hidden_channels, latent_dim, output_shape):
        """

        """
        super().__init__()

        self.output_shape = output_shape
        self.hidden_channels = hidden_channels

        # out features will work for images of size 28x28. 32x32 and 64x64
        # would crash for sizes that are not divisible by 4
        self.fc = nn.Linear(in_features=latent_dim,
                            out_features=hidden_channels * 2 * output_shape[1] // 4 * output_shape[2] // 4)

        self.conv2 = nn.ConvTranspose2d(in_channels=hidden_channels * 2,
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

    def forward(self, z: torch.Tensor):
        """
        :param z: input batch from latent space
        :returns: mu(x), sigma
        """
        x = self.fc(z)
        x = x.view(x.size(0), self.hidden_channels * 2, self.output_shape[1] // 4, self.output_shape[2] // 4)
        x = self.activation(self.conv2(x))
        x_mu = self.conv1(x)
        sigma = F.softplus(self.pre_sigma).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)

        return x_mu, sigma
