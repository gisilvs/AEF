from typing import List

import numpy as np
import torch
from torch import Tensor, nn

import torch.nn.functional as F
from models.autoencoder import Coder
from models.autoencoder_base import AutoEncoder
from models.vdvae import get_encoder_string, Block, parse_layer_string, get_3x3, BOTTLENECK_MULTIPLE, \
    get_decoder_string, DecBlock, get_conv


class DeterministicConvolutionalEncoderSmall(Coder):
    def __init__(self, hidden_channels: int, input_shape: List, latent_dims: int):
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

class DeterministicConvolutionalEncoderBig(Coder):
    def __init__(self, input_shape: List, latent_dims: int, size: str = None):
        super(DeterministicConvolutionalEncoderBig, self).__init__(latent_dims)
        self.input_shape = input_shape

        enc_str = get_encoder_string(input_shape, latent_dims, size)
        self.in_conv = get_3x3(input_shape[0], latent_dims)
        enc_blocks = []
        blockstr = parse_layer_string(enc_str)
        squeeze_dim = max(1, int(latent_dims * BOTTLENECK_MULTIPLE))
        for res, down_rate in blockstr[:-1]:
            use_3x3 = res > 2  # Don't use 3x3s for 1x1, 2x2 patches
            enc_blocks.append(Block(latent_dims, squeeze_dim,
                                    latent_dims, down_rate=down_rate, residual=True, use_3x3=use_3x3))

        res, down_rate = blockstr[-1]
        use_3x3 = res > 2
        enc_blocks.append(Block(latent_dims, squeeze_dim,
                                latent_dims, down_rate=down_rate, residual=False, use_3x3=use_3x3))
        n_blocks = len(blockstr)
        for b in enc_blocks:
            b.c4.weight.data *= np.sqrt(1 / n_blocks)
        self.enc_blocks = nn.ModuleList(enc_blocks)

    def forward(self, x):
        x = x.contiguous()
        x = self.in_conv(x)
        for block in self.enc_blocks:
            x = block(x)

        x = x.view(x.shape[0], -1)
        return x


class DeterministicConvolutionalDecoderBig(Coder):
    def __init__(self, output_shape: List, latent_dims: int, size: str = None):
        super(DeterministicConvolutionalDecoderBig, self).__init__(latent_dims)
        self.output_shape = output_shape

        dec_blocks = []
        dec_str = get_decoder_string(output_shape, latent_dims, size)
        blocks = parse_layer_string(dec_str)
        for idx, (res, mixin) in enumerate(blocks):
            dec_blocks.append(DecBlock(res, mixin, n_blocks=len(blocks), width=latent_dims))
        self.dec_blocks = nn.ModuleList(dec_blocks)
        self.gain = nn.Parameter(torch.ones(1, latent_dims, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, latent_dims, 1, 1))
        self.final_fn = lambda x: x * self.gain + self.bias
        self.out_conv = get_conv(latent_dims, output_shape[0], kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = x.view(x.shape[0], x.shape[1], 1, 1)
        for block in self.dec_blocks:
            x = block(x)
        x = self.final_fn(x)
        x = self.out_conv(x)
        return x


class StandardAutoEncoder(AutoEncoder):
    def __init__(self, encoder: DeterministicConvolutionalEncoderSmall, decoder: DeterministicConvolutionalDecoderSmall,
                 preprocessing_layers: List = ()):
        super(StandardAutoEncoder, self).__init__(encoder, decoder)
        self.preprocessing_layers = nn.ModuleList(preprocessing_layers)

    def encode(self, x: Tensor):
        for layer in self.preprocessing_layers:
            x, _ = layer.inverse(x)
        return self.encoder(x)

    def decode(self, z: Tensor):
        x = self.decoder(z)
        for i in range(len(self.preprocessing_layers) - 1, -1, -1):
            x, _ = self.preprocessing_layers[i](x)
        return x

    def forward(self, x: Tensor):
        z = self.encode(x)
        x_reconstructed = self.decode(z)
        return x_reconstructed

    def loss_function(self, x: Tensor):
        x_reconstructed = self.forward(x)
        batch_loss = F.mse_loss(x_reconstructed, x, reduction='none')
        return batch_loss

