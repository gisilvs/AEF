from typing import List
from collections import defaultdict
import torch
from torch import nn
from torch.nn import functional as F
from torch import Tensor, distributions
import numpy as np
import itertools
from util import count_parameters

WIDTH = 256 #384
DEC_BLOCKS = "1x1,4m1,4x2,8m4,8x5,16m8,16x10,32m16,32x21"
ENC_BLOCKS = "32x11,32d2,16x6,16d2,8x6,8d2,4x3,4d4,1x3"
BOTTLENECK_MULTIPLE = 0.25
CUSTOM_WIDTH_STR = ''
IMAGE_CHANNELS = 3


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


@torch.jit.script
def gaussian_analytical_kl(mu1, mu2, logsigma1, logsigma2):
    return -0.5 + logsigma2 - logsigma1 + 0.5 * (logsigma1.exp() ** 2 + (mu1 - mu2) ** 2) / (logsigma2.exp() ** 2)

@torch.jit.script
def draw_gaussian_diag_samples(mu, logsigma):
    eps = torch.empty_like(mu).normal_(0., 1.)
    return torch.exp(logsigma) * eps + mu

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

class DecBlock(nn.Module):
    def __init__(self, res, mixin, n_blocks):
        super().__init__()
        self.base = res
        self.mixin = mixin
        self.widths = get_width_settings(width=WIDTH, s=CUSTOM_WIDTH_STR)
        width = self.widths[res]
        use_3x3 = res > 2
        cond_width = int(width * BOTTLENECK_MULTIPLE)
        self.resnet = Block(width, cond_width, width, residual=True, use_3x3=use_3x3)
        self.resnet.c4.weight.data *= np.sqrt(1 / n_blocks)

    def forward(self, x):
        if self.mixin is not None:
            x = F.interpolate(x[:, :x.shape[1], ...], scale_factor=self.base // self.mixin)
        x = self.resnet(x)
        return x


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.build()
    def build(self):
        self.in_conv = get_3x3(IMAGE_CHANNELS, WIDTH)
        self.widths = get_width_settings(WIDTH, CUSTOM_WIDTH_STR)
        enc_blocks = []
        blockstr = parse_layer_string(ENC_BLOCKS)
        for res, down_rate in blockstr[:-1]:
            use_3x3 = res > 2  # Don't use 3x3s for 1x1, 2x2 patches
            enc_blocks.append(Block(self.widths[res], int(self.widths[res] * BOTTLENECK_MULTIPLE), self.widths[res], down_rate=down_rate, residual=True, use_3x3=use_3x3))
        res, down_rate = blockstr[-1]
        enc_blocks.append(Block(self.widths[res], int(self.widths[res] * BOTTLENECK_MULTIPLE), self.widths[res]*2, down_rate=down_rate, residual=False, use_3x3=use_3x3))
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

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.build()

    def build(self):
        dec_blocks = []
        blocks = parse_layer_string(DEC_BLOCKS)
        for idx, (res, mixin) in enumerate(blocks):
            dec_blocks.append(DecBlock(res, mixin, n_blocks=len(blocks)))
        self.dec_blocks = nn.ModuleList(dec_blocks)
        self.gain = nn.Parameter(torch.ones(1, WIDTH, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, WIDTH, 1, 1))
        self.final_fn = lambda x: x * self.gain + self.bias
        self.out_conv = get_conv(WIDTH, IMAGE_CHANNELS, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = x.view(x.shape[0], x.shape[1], 1, 1)
        for block in self.dec_blocks:
            x = block(x)
        x = self.final_fn(x)
        x = self.out_conv(x)
        return x, torch.ones_like(x)


class VDVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.build()
        self.eps = 1e-5
        self.device = None
        self.latent_dim = WIDTH
        self.prior = distributions.normal.Normal(torch.zeros(self.latent_dim), torch.ones(self.latent_dim))

    def build(self):
        self.encoder = Encoder()
        self.decoder = Decoder()

    def encode(self, x: Tensor):
        return self.encoder(x) # Encoder returns mu and log(sigma)

    def decode(self, z: Tensor):
        decoded_mu, decoded_sigma = self.decoder(z)
        return decoded_mu, decoded_sigma

    def sample(self, num_samples: int, temperature=1):
        # multiplying by the temperature works like the reparametrization trick,
        # only if the prior is N(0,1)
        z = self.prior.sample((num_samples,)).to(self.get_device()) * temperature
        return self.decode(z)[0]

    def forward(self, x: Tensor):
        z_mu, z_sigma = self.encode(x)
        z = distributions.normal.Normal(z_mu, z_sigma + self.eps).rsample().to(self.get_device())
        x_mu, x_sigma = self.decode(z)
        return x_mu, x_sigma, z_mu, z_sigma

    def loss_function(self, x: Tensor):
        x_mu, x_sigma, z_mu, z_sigma = self.forward(x)
        reconstruction_loss = torch.distributions.normal.Normal(x_mu, x_sigma + self.eps).log_prob(x).sum([1, 2, 3])
        q_z = distributions.normal.Normal(z_mu, z_sigma + self.eps)

        kl_div = distributions.kl.kl_divergence(q_z, self.prior).sum(1)
        return -(reconstruction_loss - kl_div)

    def get_device(self):
        if self.device is None:
            self.device = next(self.encoder.parameters()).device
            self.prior = distributions.normal.Normal(torch.zeros(self.latent_dim).to(self.device),
                                                     torch.ones(self.latent_dim).to(self.device))
        return self.device