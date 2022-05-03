from typing import List

from torch import Tensor, nn

import torch.nn.functional as F
from models.autoencoder import Coder
from models.autoencoder_base import AutoEncoder

class DeterministicConvolutionalEncoderSmall(Coder):
    def __init__(self, hidden_channels: int, input_shape: List, latent_dims: int):
        '''
        Default convolutional encoder class.
        :param hidden_channels:
        :param input_shape: [C,H,W]
        :param latent_dims:
        '''
        super(DeterministicConvolutionalEncoderSmall, self).__init__(latent_dims=latent_dims)
        self.input_shape = input_shape
        self.conv1 = nn.Conv2d(in_channels=input_shape[0],
                               out_channels=hidden_channels,
                               kernel_size=3,
                               stride=2,
                               padding=1)

        self.conv2 = nn.Conv2d(in_channels=hidden_channels,
                               out_channels=hidden_channels * 2,
                               kernel_size=3,
                               stride=2,
                               padding=1)

        self.fc = nn.Linear(in_features=hidden_channels * 2 * input_shape[1] // 4 * input_shape[2] // 4,
                               out_features=latent_dims)

        self.activation = nn.ReLU()

    def forward(self, x: Tensor):
        """
        :param x: batch of images with shape [batch, channels, w, h]
        :returns: mu(x), softplus(sigma(x))
        """
        batch_size = x.shape[0]
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))

        x = x.view(batch_size, -1)

        z = self.fc(x)

        return z

class DeterministicConvolutionalDecoderSmall(Coder):
    def __init__(self, hidden_channels: int, output_shape: List, latent_dims: int):
        super(DeterministicConvolutionalDecoderSmall, self).__init__(latent_dims=latent_dims)
        self.output_shape = output_shape
        self.hidden_channels = hidden_channels

        # out features will work for images of size 28x28. 32x32 and 64x64
        # would crash for sizes that are not divisible by 4
        self.fc = nn.Linear(in_features=latent_dims,
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


    def forward(self, z: Tensor):
        """
        :param z:
        :returns: mu(z), sigma(z)
        """
        x = self.fc(z)
        x = x.view(z.shape[0], self.hidden_channels * 2, self.output_shape[1] // 4, self.output_shape[2] // 4)
        x = self.activation(self.conv2(x))
        x = self.conv1(x)
        return x


class DenoisingAutoEncoder(AutoEncoder):
    def __init__(self, encoder: DeterministicConvolutionalEncoderSmall, decoder: DeterministicConvolutionalDecoderSmall):
        super(DenoisingAutoEncoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def encode(self, x: Tensor):
        return self.encoder(x)

    def decode(self, z: Tensor):
        return self.decoder(z)

    def sample(self, n_samples: int):
        raise NotImplementedError

    def forward(self, x: Tensor):
        z = self.encoder(x)
        x_reconstructed = self.decoder(z)
        return x_reconstructed, z

    def loss_function(self, x_noisy: Tensor, x_original: Tensor):
        x_reconstructed, z = self.forward(x_noisy)
        batch_loss = F.mse_loss(x_reconstructed, x_original, reduction='none')
        return batch_loss
