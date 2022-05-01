from typing import List
from collections import defaultdict
import torch
from torch import nn
from torch.nn import functional as F
from torch import Tensor, distributions
import numpy as np
import itertools

from models.autoencoder import GaussianEncoder, GaussianDecoder

# TODO: do we still need these?

#DEC_BLOCKS = "1x1,4m1,4x2,8m4,8x5,16m8,16x10,32m16,32x21"
#ENC_BLOCKS = "32x11,32d2,16x6,16d2,8x6,8d2,4x3,4d4,1x3"
BOTTLENECK_MULTIPLE = 0.25

def get_encoder_string(image_dim: List, latent_ndims: int, size: str = None):
    if image_dim == [3, 32, 32]: #and latent_ndims >= 32:
        if size == 'test':
            enc_block_str = "32x1,32d2,16x1,16d2,8x1,8d2,4x1,4d4,1x3"
        else:
            #enc_block_str = "32x11,32d2,16x6,16d2,8x6,8d2,4x3,4d4,1x3"
            enc_block_str = "32x1,32d2,16x1,16d2,8x1,8d2,4x1,4d4,1x3"
    return enc_block_str


def get_decoder_string(image_dim: List, latent_ndims: int, size: str = None):
    if image_dim == [3, 32, 32]: #and latent_ndims >= 32:
        if size == 'test':
            dec_block_str = "1x1,4m1,4x1,8m4,8x1,16m8,16x1,32m16,32x1"
        else:
            #dec_block_str = "1x1,4m1,4x2,8m4,8x5,16m8,16x10,32m16,32x21"
            dec_block_str = "1x1,4m1,4x1,8m4,8x1,16m8,16x1,32m16,32x1"
    return dec_block_str

def pad_channels(t, width):
    d1, d2, d3, d4 = t.shape
    empty = torch.zeros(d1, width, d3, d4, device=t.device)
    empty[:, :d2, :, :] = t
    return empty

def parse_layer_string(s):
    layers = []
    for ss in s.split(','):
        if 'x' in ss:
            res, num = ss.split('x')
            count = int(num)
            layers += [(int(res), None) for _ in range(count)]
        elif 'm' in ss:
            res, mixin = [int(a) for a in ss.split('m')]
            layers.append((res, mixin))
        elif 'd' in ss:
            res, down_rate = [int(a) for a in ss.split('d')]
            layers.append((res, down_rate))
        else:
            res = int(ss)
            layers.append((res, None))
    return layers


def get_conv(in_dim, out_dim, kernel_size, stride, padding, zero_bias=True, zero_weights=False, groups=1, scaled=False):
    c = nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding, groups=groups)
    if zero_bias:
        c.bias.data *= 0.0
    if zero_weights:
        c.weight.data *= 0.0
    return c

def get_width_settings(width, s):
    mapping = defaultdict(lambda: width)
    if s:
        s = s.split(',')
        for ss in s:
            k, v = ss.split(':')
            mapping[int(k)] = int(v)
    return mapping


def get_3x3(in_dim, out_dim, zero_bias=True, zero_weights=False, groups=1, scaled=False):
    return get_conv(in_dim, out_dim, 3, 1, 1, zero_bias, zero_weights, groups=groups, scaled=scaled)


def get_1x1(in_dim, out_dim, zero_bias=True, zero_weights=False, groups=1, scaled=False):
    return get_conv(in_dim, out_dim, 1, 1, 0, zero_bias, zero_weights, groups=groups, scaled=scaled)


class Block(nn.Module):
    def __init__(self, in_width, middle_width, out_width, down_rate=None, residual=False, use_3x3=True, zero_last=False):
        super().__init__()
        self.down_rate = down_rate
        self.residual = residual
        self.c1 = get_1x1(in_width, middle_width)
        self.c2 = get_3x3(middle_width, middle_width) if use_3x3 else get_1x1(middle_width, middle_width)
        self.c3 = get_3x3(middle_width, middle_width) if use_3x3 else get_1x1(middle_width, middle_width)
        self.c4 = get_1x1(middle_width, out_width, zero_weights=zero_last)

    def forward(self, x):
        xhat = self.c1(F.gelu(x))
        xhat = self.c2(F.gelu(xhat))
        xhat = self.c3(F.gelu(xhat))
        xhat = self.c4(F.gelu(xhat))
        out = x + xhat if self.residual else xhat
        if self.down_rate is not None:
            out = F.avg_pool2d(out, kernel_size=self.down_rate, stride=self.down_rate)
        return out


class ConvolutionalEncoderBig(GaussianEncoder):
    def __init__(self, input_shape: List, latent_ndims: int, size: str = None):
        super(ConvolutionalEncoderBig, self).__init__(input_shape, latent_ndims)

        enc_str = get_encoder_string(input_shape, latent_ndims, size)
        self.in_conv = get_3x3(input_shape[0], latent_ndims)
        enc_blocks = []
        blockstr = parse_layer_string(enc_str)
        squeeze_dim = max(1, int(latent_ndims * BOTTLENECK_MULTIPLE))
        for res, down_rate in blockstr[:-1]:
            use_3x3 = res > 2  # Don't use 3x3s for 1x1, 2x2 patches
            enc_blocks.append(Block(latent_ndims, squeeze_dim,
                                    latent_ndims, down_rate=down_rate, residual=True, use_3x3=use_3x3))

        res, down_rate = blockstr[-1]
        use_3x3 = res > 2
        enc_blocks.append(Block(latent_ndims, squeeze_dim,
                                latent_ndims*2, down_rate=down_rate, residual=False, use_3x3=use_3x3))
        n_blocks = len(blockstr)
        for b in enc_blocks:
            b.c4.weight.data *= np.sqrt(1 / n_blocks)
        self.enc_blocks = nn.ModuleList(enc_blocks)

    def forward(self, x):
        x = x.contiguous()
        x = self.in_conv(x)
        for block in self.enc_blocks:
            x = block(x)
        mu, sigma = torch.split(x, split_size_or_sections=x.shape[1]//2, dim=1)
        mu = mu.view(mu.shape[0], -1)
        sigma = sigma.view(sigma.shape[0], -1)
        return mu, F.softplus(sigma)


class DecBlock(nn.Module):
    def __init__(self, res, mixin, n_blocks, width):
        super().__init__()
        self.base = res
        self.mixin = mixin
        # self.widths = get_width_settings(width=width, s='') # TODO: remove
        use_3x3 = res > 2
        cond_width = max(1, int(width * BOTTLENECK_MULTIPLE))
        self.resnet = Block(width, cond_width, width, residual=True, use_3x3=use_3x3)
        self.resnet.c4.weight.data *= np.sqrt(1 / n_blocks)

    def forward(self, x):
        if self.mixin is not None:
            x = F.interpolate(x[:, :x.shape[1], ...], scale_factor=self.base // self.mixin)
        x = self.resnet(x)
        return x


class ConvolutionalDecoderBig(GaussianDecoder):
    def __init__(self, output_shape: List, latent_ndims: int, size: str = None):
        super(ConvolutionalDecoderBig, self).__init__(output_shape, latent_ndims)

        dec_blocks = []
        dec_str = get_decoder_string(output_shape, latent_ndims, size)
        blocks = parse_layer_string(dec_str)
        for idx, (res, mixin) in enumerate(blocks):
            dec_blocks.append(DecBlock(res, mixin, n_blocks=len(blocks), width=latent_ndims))
        self.dec_blocks = nn.ModuleList(dec_blocks)
        self.gain = nn.Parameter(torch.ones(1, latent_ndims, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, latent_ndims, 1, 1))
        self.final_fn = lambda x: x * self.gain + self.bias
        self.out_conv = get_conv(latent_ndims, output_shape[0], kernel_size=1, stride=1, padding=0)


class FixedVarianceDecoderBig(ConvolutionalDecoderBig):
    def __init__(self, output_shape: List, latent_ndims: int, size: str = None):
        super(FixedVarianceDecoderBig, self).__init__(output_shape, latent_ndims, size)

    def forward(self, x):
        x = x.view(x.shape[0], x.shape[1], 1, 1)
        for block in self.dec_blocks:
            x = block(x)
        x = self.final_fn(x)
        x = self.out_conv(x)
        return x, torch.ones_like(x)


class IndependentVarianceDecoderBig(ConvolutionalDecoderBig):
    def __init__(self, output_shape: List, latent_ndims: int, size: str = None):
        super(IndependentVarianceDecoderBig, self).__init__(output_shape, latent_ndims, size)
        self.pre_sigma = nn.Parameter(torch.ones(output_shape))

    def forward(self, z):
        x = z.view(z.shape[0], z.shape[1], 1, 1)
        for block in self.dec_blocks:
            x = block(x)
        x = self.final_fn(x)
        x_mu = self.out_conv(x)
        sigma = F.softplus(self.pre_sigma).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
        return x_mu, sigma


class LatentDependentDecoderBig(GaussianDecoder):
    def __init__(self, output_shape: List, latent_ndims: int, size: str = None):
        super(LatentDependentDecoderBig, self).__init__(output_shape, latent_ndims)

        dec_blocks = []
        dec_str = get_decoder_string(output_shape, latent_ndims, size)
        blocks = parse_layer_string(dec_str)
        for idx, (res, mixin) in enumerate(blocks):
            dec_blocks.append(DecBlock(res, mixin, n_blocks=len(blocks), width=latent_ndims))

        self.dec_blocks = nn.ModuleList(dec_blocks)
        self.gain = nn.Parameter(torch.ones(1, latent_ndims, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, latent_ndims, 1, 1))
        self.final_fn = lambda x: x * self.gain + self.bias
        self.out_conv = get_conv(latent_ndims, output_shape[0]*2, kernel_size=1, stride=1, padding=0)

    def forward(self, z):
        x = z.view(z.shape[0], z.shape[1], 1, 1)
        for block in self.dec_blocks:
            x = block(x)
        x = self.final_fn(x)
        x = self.out_conv(x)
        x_mu, x_sigma = torch.split(x, split_size_or_sections=x.shape[1] // 2, dim=1)
        # x_mu = x_mu.view(x_mu.shape[0], -1)
        # x_sigma = x_sigma.view(x_sigma.shape[0], -1)
        return x_mu, F.softplus(x_sigma)


# "1x1,4m1,4x2,8m4,8x4,14m8,14x6,28m14,28x14"