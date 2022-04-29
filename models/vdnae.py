from typing import List
from collections import defaultdict
import torch
from torch import nn
from torch.nn import functional as F
from torch import Tensor, distributions
import numpy as np
import itertools
from util import count_parameters
from nflows.transforms import InverseTransform, AffineTransform

from bijectors.actnorm import ActNorm
from bijectors.sigmoid import Sigmoid
from bijectors.masked_autoregressive_transform import get_masked_autoregressive_transform
from torch.distributions.normal import Normal

WIDTH = 512 #384
lr = 0.0002
ZDIM = 16
wd = 0.01
DEC_BLOCKS = "1x1,4m1,4x2,8m4,8x5,16m8,16x10,32m16,32x21"
ENC_BLOCKS = "32x11,32d2,16x6,16d2,8x6,8d2,4x3,4d4,1x3"
warmup_iters = 100
dataset = 'cifar10'
n_batch = 16
ema_rate = 0.9999
BOTTLENECK_MULTIPLE = 0.25
CUSTOM_WIDTH_STR = ''
IMAGE_SIZE = 32
IMAGE_CHANNELS = 3
NO_BIAS_ABOVE = 64


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
        self.out_conv = get_conv(WIDTH, 3, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        for block in self.dec_blocks:
            x = block(x)
        x = self.final_fn(x)
        x = self.out_conv(x)
        return x, torch.ones_like(x)


class VDNAE(nn.Module):
    def __init__(self, hardcoded_mask=True, core_flow_fn=get_masked_autoregressive_transform):
        super(VDNAE, self).__init__()

        self.preprocessing_layers = preprocessing_layers = [InverseTransform(AffineTransform(0.05, 1 - 2 * 0.05)), Sigmoid(), ActNorm(3)]
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.core_size = WIDTH
        self.image_shape = [3,32,32]
        self.core_flow_pre = core_flow_fn(features=self.core_size,
                                                  hidden_features=512,
                                                  num_layers=8,
                                                  num_blocks_per_layer=2,
                                                  act_norm_between_layers=True)
        self.core_flow_post = core_flow_fn(features=self.core_size,
                                                   hidden_features=512,
                                                   num_layers=8,
                                                   num_blocks_per_layer=2,
                                                   act_norm_between_layers=True)
        self.eps = 1e-5

        preprocessing_layers = nn.ModuleList(preprocessing_layers)
        self.preprocessing_layers = preprocessing_layers
        self.device = None

        if hardcoded_mask:
            mask = torch.zeros(self.image_shape)
            if self.core_size == 2:
                mask[0, 13:15, 13] = 1
            elif self.core_size == 4:
                mask[0, 13:15, 13:15] = 1
            elif self.core_size == 8:
                mask[0, 12:16, 13:15] = 1
            elif self.core_size == 16:
                mask[0, 12:16, 12:16] = 1
            elif self.core_size == 32:
                mask[0, 10:18, 12:16] = 1
            elif self.core_size == 256:
                mask[0, 8:24, 8:24] = 1
            elif self.core_size == 512:
                mask[0, 8:24, 8:24] = 1
                mask[1, 8:24, 8:24] = 1
            else:
                print('NOT IMPLEMENTED YET')
                exit(1)
            self.register_buffer('mask', mask)
            #self.mask = mask
        else:
            mask = self._get_mask()
            self.register_buffer('mask', mask)
            #self.mask = self._get_mask()

    def _get_mask(self):
        '''
        simple procedure to fill up first each corner of each channel, and then proceeding along the sides.
        plot the mask for a visual understanding
        :return:
        '''
        mask = torch.zeros(self.image_shape)
        width = self.image_shape[1]
        height = self.image_shape[2]
        n_channels = self.image_shape[0]
        counter = 0
        row = 0
        column = 0
        channel = 0
        base_number_cols = 0
        base_number_rows = 0
        while (1):
            mask[channel, row, column] = 1
            counter += 1
            if counter == self.core_size:
                break
            mask[channel, row, height - column - 1] = 1
            counter += 1
            if counter == self.core_size:
                break
            mask[channel, width - row - 1, column] = 1
            counter += 1
            if counter == self.core_size:
                break
            mask[channel, width - row - 1, height - column - 1] = 1
            counter += 1
            if counter == self.core_size:
                break
            channel += 1
            if channel == n_channels:
                channel = 0
                if row == column:
                    row += 1
                elif row > column:
                    column = row
                    row = base_number_rows
                    if column > height // 2:
                        base_number_cols += 1
                        column = 0
                elif column > row:
                    row = column + 1
                    column = base_number_cols
                    if row > width // 2:
                        base_number_rows += 1
                        base_number_cols += 1
                        row = 0
        return mask

    def partition(self, x):
        core = x[:, self.mask == 1].view(x.shape[0], -1)
        shell = x * (1 - self.mask)
        return core, shell

    def inverse_partition(self, core, shell):
        shell[:, self.mask == 1] = core
        return shell

    def embedding(self, x):
        log_j_preprocessing = 0
        for layer in self.preprocessing_layers:
            x, log_j_transform = layer.inverse(x)
            log_j_preprocessing += log_j_transform

        core, shell = self.partition(x)
        core, log_j_core_pre = self.core_flow_pre.inverse(core)
        mu_z, sigma_z = self.encoder(shell)
        z = (core - mu_z.view(-1, self.core_size)) / (sigma_z.view(-1, self.core_size) + self.eps)
        log_j_z = torch.sum(-torch.log(sigma_z.view(-1, self.core_size) + self.eps), dim=[1])
        mu_d, sigma_d = self.decoder(z.view(-1, self.core_size, 1,1))
        deviations = (shell - mu_d) / (sigma_d + self.eps)
        log_j_d = torch.sum(-torch.log(sigma_d[:, self.mask == 0] + self.eps),
                            dim=[1])
        z, log_j_core_post = self.core_flow_post.inverse(z)
        return z, deviations, log_j_preprocessing + log_j_core_pre + log_j_z + log_j_d + log_j_core_post

    def neg_log_likelihood(self, x):
        z, deviations, log_j = self.embedding(x)
        loss_z = torch.sum(
            Normal(loc=torch.zeros_like(z), scale=torch.ones_like(z)).log_prob(
                z), dim=1)
        loss_d = torch.sum(Normal(loc=torch.zeros_like(deviations[:, self.mask == 0]),
                                  scale=torch.ones_like(deviations[:, self.mask == 0])).log_prob(
            deviations[:, self.mask == 0]), dim=[1])
        return -(loss_z + loss_d + log_j)

    def encode(self, x: Tensor):
        z, deviations, _ = self.embedding(x)
        return z, deviations

    def decode(self, z: Tensor, deviations: Tensor):
        core, shell = self.forward(z, deviations)
        x = self.inverse_partition(core, shell)
        return x

    def sample(self, num_samples=1, sample_deviations=False, temperature=1.):
        device = self.get_device()
        z = torch.normal(torch.zeros(num_samples, self.core_size),
                         torch.ones(num_samples, self.core_size)*temperature).to(device)
        if sample_deviations:
            deviations = torch.normal(torch.zeros_like(self.mask),
                                      torch.ones_like(self.mask)*temperature).to(device)
        else:
            deviations = None

        core, shell = self.forward(z, deviations)
        y = self.inverse_partition(core, shell)
        for i in range(len(self.preprocessing_layers) - 1, -1, -1):
            y, _ = self.preprocessing_layers[i](y)
        return y

    def forward(self, z, deviations=None):
        z, _ = self.core_flow_post.forward(z)
        mu_d, sigma_d = self.decoder(z.view(-1, self.core_size, 1,1))
        if deviations is None:
            shell = mu_d
        else:
            shell = deviations * (sigma_d + self.eps) + mu_d
        shell = shell * (1 - self.mask)
        mu_z, sigma_z = self.encoder(shell)
        core = z * (sigma_z.view(-1, self.core_size) + self.eps) + mu_z.view(-1, self.core_size)
        core, _ = self.core_flow_pre.forward(core)
        return core, shell

    def loss_function(self, x: Tensor):
        return self.neg_log_likelihood(x)

    def get_device(self):
        if self.device is None:
            self.device = next(self.parameters()).device
            #self.mask = self.mask.to(self.device)
        return self.device

'''model = VDNAE()
model.loss_function(model.sample(2))'''