from collections import defaultdict
from ast import literal_eval
import itertools
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from torch.distributions.normal import Normal

from utils.hvae_utils import get_1x1, get_3x3, draw_gaussian_diag_samples, \
    gaussian_analytical_kl, get_conv

BOTTLENECK_MULTIPLE = 0.5


def log_prob_from_logits(x):
    """ numerically stable log_softmax implementation that prevents overflow """
    axis = len(x.shape) - 1
    m = x.max(dim=axis, keepdim=True)[0]
    return x - m - torch.log(torch.exp(x - m).sum(dim=axis, keepdim=True))


def const_max(t, constant):
    other = torch.ones_like(t) * constant
    return torch.max(t, other)


def const_min(t, constant):
    other = torch.ones_like(t) * constant
    return torch.min(t, other)


def discretized_mix_logistic_loss(x, l, low_bit=False):
    """ log-likelihood for mixture of discretized logistics, assumes the data has been rescaled to [-1,1] interval """
    # Adapted from https://github.com/openai/pixel-cnn/blob/master/pixel_cnn_pp/nn.py
    xs = [s for s in
          x.shape]  # true image (i.e. labels) to regress to, e.g. (B,32,32,3)
    ls = [s for s in l.shape]  # predicted distribution, e.g. (B,32,32,100)
    nr_mix = int(ls[
                     -1] / 10)  # here and below: unpacking the params of the mixture of logistics
    logit_probs = l[:, :, :, :nr_mix]
    l = torch.reshape(l[:, :, :, nr_mix:], xs + [nr_mix * 3])
    means = l[:, :, :, :, :nr_mix]
    log_scales = const_max(l[:, :, :, :, nr_mix:2 * nr_mix], -7.)
    coeffs = torch.tanh(l[:, :, :, :, 2 * nr_mix:3 * nr_mix])
    x = torch.reshape(x, xs + [1]) + torch.zeros(xs + [nr_mix]).to(
        x.device)  # here and below: getting the means and adjusting them based on preceding sub-pixels
    m2 = torch.reshape(
        means[:, :, :, 1, :] + coeffs[:, :, :, 0, :] * x[:, :, :, 0, :],
        [xs[0], xs[1], xs[2], 1, nr_mix])
    m3 = torch.reshape(
        means[:, :, :, 2, :] + coeffs[:, :, :, 1, :] * x[:, :, :, 0,
                                                       :] + coeffs[:, :, :, 2,
                                                            :] * x[:, :, :, 1,
                                                                 :],
        [xs[0], xs[1], xs[2], 1, nr_mix])
    means = torch.cat(
        [torch.reshape(means[:, :, :, 0, :], [xs[0], xs[1], xs[2], 1, nr_mix]),
         m2, m3], dim=3)
    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    if low_bit:
        plus_in = inv_stdv * (centered_x + 1. / 31.)
        cdf_plus = torch.sigmoid(plus_in)
        min_in = inv_stdv * (centered_x - 1. / 31.)
    else:
        plus_in = inv_stdv * (centered_x + 1. / 255.)
        cdf_plus = torch.sigmoid(plus_in)
        min_in = inv_stdv * (centered_x - 1. / 255.)
    cdf_min = torch.sigmoid(min_in)
    log_cdf_plus = plus_in - F.softplus(
        plus_in)  # log probability for edge case of 0 (before scaling)
    log_one_minus_cdf_min = -F.softplus(
        min_in)  # log probability for edge case of 255 (before scaling)
    cdf_delta = cdf_plus - cdf_min  # probability for all other cases
    mid_in = inv_stdv * centered_x
    log_pdf_mid = mid_in - log_scales - 2. * F.softplus(
        mid_in)  # log probability in the center of the bin, to be used in extreme cases (not actually used in our code)

    # now select the right output: left edge case, right edge case, normal case, extremely low prob case (doesn't actually happen for us)

    # this is what we are really doing, but using the robust version below for extreme cases in other applications and to avoid NaN issue with tf.select()
    # log_probs = tf.select(x < -0.999, log_cdf_plus, tf.select(x > 0.999, log_one_minus_cdf_min, tf.log(cdf_delta)))

    # robust version, that still works if probabilities are below 1e-5 (which never happens in our code)
    # tensorflow backpropagates through tf.select() by multiplying with zero instead of selecting: this requires use to use some ugly tricks to avoid potential NaNs
    # the 1e-12 in tf.maximum(cdf_delta, 1e-12) is never actually used as output, it's purely there to get around the tf.select() gradient issue
    # if the probability on a sub-pixel is below 1e-5, we use an approximation based on the assumption that the log-density is constant in the bin of the observed sub-pixel value
    if low_bit:
        log_probs = torch.where(x < -0.999,
                                log_cdf_plus,
                                torch.where(x > 0.999,
                                            log_one_minus_cdf_min,
                                            torch.where(cdf_delta > 1e-5,
                                                        torch.log(
                                                            const_max(cdf_delta,
                                                                      1e-12)),
                                                        log_pdf_mid - np.log(
                                                            15.5))))
    else:
        log_probs = torch.where(x < -0.999,
                                log_cdf_plus,
                                torch.where(x > 0.999,
                                            log_one_minus_cdf_min,
                                            torch.where(cdf_delta > 1e-5,
                                                        torch.log(
                                                            const_max(cdf_delta,
                                                                      1e-12)),
                                                        log_pdf_mid - np.log(
                                                            127.5))))
    log_probs = log_probs.sum(dim=3) + log_prob_from_logits(logit_probs)
    mixture_probs = torch.logsumexp(log_probs, -1)
    return -1. * mixture_probs.sum(dim=[1, 2]) / np.prod(xs[1:])


def sample_from_discretized_mix_logistic(l, nr_mix):
    ls = [s for s in l.shape]
    xs = ls[:-1] + [3]
    # unpack parameters
    logit_probs = l[:, :, :, :nr_mix]
    l = torch.reshape(l[:, :, :, nr_mix:], xs + [nr_mix * 3])
    # sample mixture indicator from softmax
    eps = torch.empty(logit_probs.shape, device=l.device).uniform_(1e-5,
                                                                   1. - 1e-5)
    amax = torch.argmax(logit_probs - torch.log(-torch.log(eps)), dim=3)
    sel = F.one_hot(amax, num_classes=nr_mix).float()
    sel = torch.reshape(sel, xs[:-1] + [1, nr_mix])
    # select logistic parameters
    means = (l[:, :, :, :, :nr_mix] * sel).sum(dim=4)
    log_scales = const_max((l[:, :, :, :, nr_mix:nr_mix * 2] * sel).sum(dim=4),
                           -7.)
    coeffs = (torch.tanh(l[:, :, :, :, nr_mix * 2:nr_mix * 3]) * sel).sum(dim=4)
    # sample from logistic & clip to interval
    # we don't actually round to the nearest 8bit value when sampling
    u = torch.empty(means.shape, device=means.device).uniform_(1e-5, 1. - 1e-5)
    x = means + torch.exp(log_scales) * (torch.log(u) - torch.log(1. - u))
    x0 = const_min(const_max(x[:, :, :, 0], -1.), 1.)
    x1 = const_min(const_max(x[:, :, :, 1] + coeffs[:, :, :, 0] * x0, -1.), 1.)
    x2 = const_min(const_max(
        x[:, :, :, 2] + coeffs[:, :, :, 1] * x0 + coeffs[:, :, :, 2] * x1, -1.),
                   1.)
    return torch.cat(
        [torch.reshape(x0, xs[:-1] + [1]), torch.reshape(x1, xs[:-1] + [1]),
         torch.reshape(x2, xs[:-1] + [1])], dim=3)


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


def get_count_per_res(s):
    count_list = []
    res_list = []
    for ss in s.split(','):
        if 'x' in ss:
            res, num = ss.split('x')
            res = int(res)
            count = int(num)
            if res > 1:
                count += 1
            count_list.append(count)
            res_list.append(res)
    return res_list, count_list


def pad_channels(t, width):
    d1, d2, d3, d4 = t.shape
    empty = torch.zeros(d1, width, d3, d4, device=t.device)
    empty[:, :d2, :, :] = t
    return empty


def get_width_settings(width, s):
    mapping = defaultdict(lambda: width)
    if s:
        s = s.split(',')
        for ss in s:
            k, v = ss.split(':')
            mapping[int(k)] = int(v)
    return mapping


class DmolNet(nn.Module):
    def __init__(self, width, num_mixtures):
        super().__init__()
        self.width = width
        self.num_mixtures = num_mixtures
        self.out_conv = get_conv(width, num_mixtures * 10, kernel_size=1,
                                 stride=1, padding=0)

    def nll(self, px_z, x):
        return discretized_mix_logistic_loss(x=x, l=self.forward(px_z))

    def forward(self, px_z):
        xhat = self.out_conv(px_z)
        return xhat.permute(0, 2, 3, 1)

    def sample(self, px_z):
        im = sample_from_discretized_mix_logistic(self.forward(px_z),
                                                  self.num_mixtures).permute(0,
                                                                             3,
                                                                             1,
                                                                             2)
        xhat = (im + 1.0) / 2.
        '''xhat = xhat.detach().cpu().numpy()
        xhat = np.minimum(np.maximum(0.0, xhat), 255.0)
        return np.transpose(xhat, (0, 3, 1, 2))'''
        return xhat


class Block(nn.Module):
    def __init__(self, in_width, middle_width, out_width, down_rate=None,
                 residual=False, use_3x3=True, zero_last=False):
        super().__init__()
        self.down_rate = down_rate
        self.residual = residual
        self.c1 = get_1x1(in_width, middle_width)
        self.c2 = get_3x3(middle_width, middle_width) if use_3x3 else get_1x1(
            middle_width, middle_width)
        self.c3 = get_3x3(middle_width, middle_width) if use_3x3 else get_1x1(
            middle_width, middle_width)
        self.c4 = get_1x1(middle_width, out_width, zero_weights=zero_last)

    def forward(self, x):
        xhat = self.c1(F.gelu(x))
        xhat = self.c2(F.gelu(xhat))
        xhat = self.c3(F.gelu(xhat))
        xhat = self.c4(F.gelu(xhat))
        out = x + xhat if self.residual else xhat
        if self.down_rate is not None:
            out = F.avg_pool2d(out, kernel_size=self.down_rate,
                               stride=self.down_rate)
        return out


class Encoder(nn.Module):

    def __init__(self,
                 image_channels,
                 width,
                 custom_width_str,
                 encoder_blocks):
        super().__init__()

        self.in_conv = get_3x3(image_channels, width)
        self.widths = get_width_settings(width, custom_width_str)
        enc_blocks = []
        blockstr = parse_layer_string(encoder_blocks)
        for res, down_rate in blockstr:
            use_3x3 = res > 2  # Don't use 3x3s for 1x1, 2x2 patches
            enc_blocks.append(Block(self.widths[res],
                                    int(self.widths[res] * BOTTLENECK_MULTIPLE),
                                    self.widths[res], down_rate=down_rate,
                                    residual=True, use_3x3=use_3x3))
        n_blocks = len(blockstr)
        for b in enc_blocks:
            b.c4.weight.data *= np.sqrt(1 / n_blocks)
        self.enc_blocks = nn.ModuleList(enc_blocks)

    def forward(self, x):
        x = x.contiguous()
        x = self.in_conv(x)
        activations = {}
        activations[x.shape[2]] = x
        for block in self.enc_blocks:
            x = block(x)
            res = x.shape[2]
            x = x if x.shape[1] == self.widths[res] else pad_channels(x,
                                                                      self.widths[
                                                                          res])
            activations[res] = x
        return activations


class DecBlock(nn.Module):
    def __init__(self, res, mixin, n_blocks, width,
                 custom_width_str,
                 z_dim,
                 is_stochastic=True):
        super().__init__()
        self.is_stochastic = is_stochastic
        self.base = res
        self.mixin = mixin
        self.widths = get_width_settings(width, custom_width_str)
        width = self.widths[res]
        use_3x3 = res > 2
        cond_width = int(width * BOTTLENECK_MULTIPLE)
        if is_stochastic:
            self.zdim = z_dim[res]
            self.enc = Block(width * 2, cond_width, z_dim[res] * 2,
                             residual=False, use_3x3=use_3x3)
            self.prior = Block(width, cond_width, z_dim[res] * 2 + width,
                               residual=False, use_3x3=use_3x3, zero_last=True)
            self.z_proj = get_1x1(z_dim[res], width)
            self.z_proj.weight.data *= np.sqrt(1 / n_blocks)
            self.z_fn = lambda x: self.z_proj(x)
            self.resnet = Block(width, cond_width, width, residual=True,
                                use_3x3=use_3x3)
            self.resnet.c4.weight.data *= np.sqrt(1 / n_blocks)
        else:
            resnet1 = Block(width, cond_width, width, residual=True,
                            use_3x3=use_3x3)
            resnet1.c4.weight.data *= np.sqrt(1 / n_blocks)
            resnet2 = Block(width, cond_width, width, residual=True,
                            use_3x3=use_3x3)
            resnet2.c4.weight.data *= np.sqrt(1 / n_blocks)
            self.resnet = nn.Sequential(resnet1, resnet2)

    def sample(self, x, acts, pre_latent=None):
        batch_size = x.shape[0]
        qm, qv = self.enc(torch.cat([x, acts], dim=1)).chunk(2, dim=1)
        feats = self.prior(x)
        pm, pv, xpp = feats[:, :self.zdim, ...], feats[:,
                                                 self.zdim:self.zdim * 2,
                                                 ...], feats[:, self.zdim * 2:,
                                                       ...]
        x = x + xpp
        if not torch.is_tensor(pre_latent):
            z = draw_gaussian_diag_samples(qm, qv)
            kl = gaussian_analytical_kl(qm, pm, qv, pv)
            return z, x, kl
        else:
            z = qm + qv.exp() * pre_latent.view(qm.shape)
            log_j_z = torch.sum(torch.log(qv.exp()).view(batch_size, -1),
                                dim=[1])
            prior_loss = torch.sum(Normal(pm, pv.exp()).log_prob(z),
                                   dim=[1, 2, 3])
            return z, x, log_j_z + prior_loss

    def sample_uncond(self, x, t=None, lvs=None):
        n, c, h, w = x.shape
        feats = self.prior(x)
        pm, pv, xpp = feats[:, :self.zdim, ...], feats[:,
                                                 self.zdim:self.zdim * 2,
                                                 ...], feats[:, self.zdim * 2:,
                                                       ...]
        x = x + xpp
        if lvs is not None:
            z = lvs
        else:
            if t is not None:
                pv = pv + torch.ones_like(pv) * np.log(t)
            z = draw_gaussian_diag_samples(pm, pv)
        return z, x

    def get_inputs(self, xs, activations):
        acts = activations[self.base]
        try:
            x = xs[self.base]
        except KeyError:
            x = torch.zeros_like(acts)
        if acts.shape[0] != x.shape[0]:
            x = x.repeat(acts.shape[0], 1, 1, 1)
        return x, acts

    def forward(self, xs, activations, get_latents=False, pre_latent=None):
        x, acts = self.get_inputs(xs, activations)
        if self.mixin is not None:
            x = x + F.interpolate(xs[self.mixin][:, :x.shape[1], ...],
                                  scale_factor=self.base // self.mixin)
        if self.is_stochastic:
            z, x, kl = self.sample(x, acts, pre_latent)
            x = x + self.z_fn(z)
        x = self.resnet(x)
        xs[self.base] = x
        if get_latents and self.is_stochastic:
            if torch.is_tensor(pre_latent):
                return xs, dict(z=z, z_loss=kl)
            else:
                return xs, dict(z=z.detach(), kl=kl)
        elif self.is_stochastic:
            return xs, dict(kl=kl)
        else:
            return xs, dict()

    def forward_uncond(self, xs, t=None, lvs=None):
        try:
            x = xs[self.base]
        except KeyError:
            ref = xs[list(xs.keys())[0]]
            x = torch.zeros(dtype=ref.dtype, size=(
            ref.shape[0], self.widths[self.base], self.base, self.base),
                            device=ref.device)
        if self.mixin is not None:
            x = x + F.interpolate(xs[self.mixin][:, :x.shape[1], ...],
                                  scale_factor=self.base // self.mixin)
        if self.is_stochastic:
            z, x = self.sample_uncond(x, t, lvs=lvs)
            x = x + self.z_fn(z)
        x = self.resnet(x)
        xs[self.base] = x
        return xs


class Decoder(nn.Module):

    def __init__(self,
                 width,
                 image_size,
                 n_channels,
                 custom_width_str,
                 decoder_blocks,
                 no_bias_above,
                 one_stochastic_per_dim,
                 z_dim,
                 num_mixtures,
                 gaussian_nll):
        super().__init__()
        resos = set()
        dec_blocks = []
        self.image_size = image_size
        self.widths = get_width_settings(width, custom_width_str)
        blocks = parse_layer_string(decoder_blocks)
        old_res = 0
        for idx, (res, mixin) in enumerate(blocks):
            if one_stochastic_per_dim:
                if res != old_res and res in z_dim.keys():
                    dec_blocks.append(
                        DecBlock(res, mixin, len(blocks), width,
                                 custom_width_str, z_dim, is_stochastic=True))
                else:
                    dec_blocks.append(
                        DecBlock(res, mixin, len(blocks), width,
                                 custom_width_str, z_dim, is_stochastic=False))
            else:
                dec_blocks.append(
                    DecBlock(res, mixin, len(blocks), width,
                             custom_width_str, z_dim, is_stochastic=True))
            old_res = res
            resos.add(res)
        self.resolutions = sorted(resos)
        self.dec_blocks = nn.ModuleList(dec_blocks)
        self.bias_xs = nn.ParameterList(
            [nn.Parameter(torch.zeros(1, self.widths[res], res, res)) for res in
             self.resolutions if res <= no_bias_above])
        if gaussian_nll:
            self.out_net = get_conv(width, n_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.out_net = DmolNet(width=width, num_mixtures=num_mixtures)
        self.gain = nn.Parameter(torch.ones(1, width, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, width, 1, 1))
        self.final_fn = lambda x: x * self.gain + self.bias

    def forward(self, activations, get_latents=False, extended_features=[]):
        stats = []
        xs = {a.shape[2]: a for a in self.bias_xs}
        b = 0
        for block in self.dec_blocks:
            if extended_features and block.is_stochastic:
                xs, block_stats = block(xs, activations,
                                        get_latents=get_latents,
                                        pre_latent=extended_features[b])
                b += 1
            else:
                xs, block_stats = block(xs, activations,
                                        get_latents=get_latents)
            stats.append(block_stats)
        xs[self.image_size] = self.final_fn(xs[self.image_size])
        return xs[self.image_size], stats

    def forward_uncond(self, n, t=None, y=None):
        xs = {}
        for bias in self.bias_xs:
            xs[bias.shape[2]] = bias.repeat(n, 1, 1, 1)
        for idx, block in enumerate(self.dec_blocks):
            try:
                temp = t[idx]
            except TypeError:
                temp = t
            xs = block.forward_uncond(xs, temp)
        xs[self.image_size] = self.final_fn(xs[self.image_size])
        return xs[self.image_size]

    def forward_manual_latents(self, n, latents, t=None):
        xs = {}
        for bias in self.bias_xs:
            xs[bias.shape[2]] = bias.repeat(n, 1, 1, 1)
        for block, lvs in itertools.zip_longest(self.dec_blocks, latents):
            xs = block.forward_uncond(xs, t, lvs=lvs)
        xs[self.image_size] = self.final_fn(xs[self.image_size])
        return xs[self.image_size]


class HVAE(nn.Module):
    def __init__(
            self,
            width=64,
            image_size=64,
            image_channels=3,
            custom_width_str="",
            encoder_blocks="64x4,64d2,32x4,32d2,16x4,16d2,8x4,8d2,4x4,4d4,1x4",
            decoder_blocks="1x4,4m1,4x4,8m4,8x4,16m8,16x4,32m16,32x4,64m32,64x4",
            z_dim=None,
            gaussian_nll=False,
            num_mixtures=10,
            no_bias_above=64,
            one_stochastic_per_dim=False):
        super().__init__()
        z_dim = literal_eval(z_dim)
        self.encoder = Encoder(image_channels,
                               width,
                               custom_width_str,
                               encoder_blocks)
        self.decoder = Decoder(width,
                               image_size,
                               image_channels,
                               custom_width_str,
                               decoder_blocks,
                               no_bias_above,
                               one_stochastic_per_dim,
                               z_dim,
                               num_mixtures,
                               gaussian_nll)
        self.gaussian_nll = gaussian_nll
        if gaussian_nll:
            self.sigma = nn.Parameter(
                torch.zeros([image_channels, image_size, image_size]))

    def forward(self, x):
        activations = self.encoder.forward(x)
        px_z, stats = self.decoder.forward(activations)
        ndims = np.prod(x.shape[1:])
        if self.gaussian_nll:
            px_z = self.decoder.out_net(px_z)
            distortion_per_pixel = -Normal(px_z,
                                          F.softplus(self.sigma)).log_prob(
                x).sum(
                [1, 2, 3]) / ndims
        else:
            distortion_per_pixel = self.decoder.out_net.nll(px_z,
                                                            x.permute(0, 2, 3,
                                                                      1))
        rate_per_pixel = torch.zeros_like(distortion_per_pixel)
        for statdict in stats:
            if 'kl' in statdict.keys():
                rate_per_pixel += statdict['kl'].sum(dim=(1, 2, 3))
        rate_per_pixel /= ndims
        elbo = (distortion_per_pixel + rate_per_pixel).mean()
        return elbo  # dict(elbo=elbo, distortion=distortion_per_pixel.mean(), rate=rate_per_pixel.mean())

    def forward_get_latents(self, x):
        activations = self.encoder.forward(x)
        _, stats = self.decoder.forward(activations, get_latents=True)
        return stats

    def forward_uncond_samples(self, n_batch, t=None):
        px_z = self.decoder.forward_uncond(n_batch, t=t)
        if self.gaussian_nll:
            return self.decoder.out_net(px_z)
        else:
            return self.decoder.out_net.sample(px_z)

    def forward_samples_set_latents(self, n_batch, latents, t=None):
        px_z = self.decoder.forward_manual_latents(n_batch, latents, t=t)
        if self.gaussian_nll:
            return self.decoder.out_net(px_z)
        else:
            return self.decoder.out_net.sample(px_z)


class HAEF(nn.Module):
    def __init__(
            self,
            width=64,
            image_size=64,
            image_channels=3,
            custom_width_str="",
            encoder_blocks="64x4,64d2,32x4,32d2,16x4,16d2,8x4,8d2,4x4,4d4,1x4",
            decoder_blocks="1x4,4m1,4x4,8m4,8x4,16m8,16x4,32m16,32x4,64m32,64x4",
            z_dim=None,
            gaussian_nll=False,
            num_mixtures=10,
            no_bias_above=64,
            one_stochastic_per_dim=False):
        super().__init__()
        z_dim = literal_eval(z_dim)
        self.res_list, self.count_list = get_count_per_res(decoder_blocks)
        self.widths = get_width_settings(width, custom_width_str)
        feat_blocks = []
        if one_stochastic_per_dim:
            self.count_list = [1 for _ in self.count_list]
        for i in range(len(self.res_list)):
            if self.res_list[i] in z_dim.keys():
                use_3x3 = True if self.res_list[i] > 2 else False
                feat_blocks.append(nn.Sequential(
                    get_1x1(3, self.widths[self.res_list[i]]),
                    Block(self.widths[self.res_list[i]],
                          int(self.widths[
                                  self.res_list[i]] * BOTTLENECK_MULTIPLE),
                          self.count_list[i] * z_dim[self.res_list[i]],
                          use_3x3=use_3x3,
                          down_rate=self.res_list[-1] // self.res_list[i])))

        self.feature_blocks = nn.ModuleList(feat_blocks)
        # self.preprocessing_layers = nn.ModuleList(preprocessing_layers)
        self.encoder = Encoder(image_channels,
                               width,
                               custom_width_str,
                               encoder_blocks)
        self.decoder = Decoder(width,
                               image_size,
                               image_channels,
                               custom_width_str,
                               decoder_blocks,
                               no_bias_above,
                               one_stochastic_per_dim,
                               z_dim,
                               num_mixtures,
                               gaussian_nll)
        self.gaussian_nll = gaussian_nll
        if gaussian_nll:
            self.sigma = nn.Parameter(
                torch.zeros([image_channels, image_size, image_size]))

    def forward(self, x):

        extended_features = []
        for i, feature_block in enumerate(self.feature_blocks):
            core = feature_block(x)
            extended_features.extend(core.chunk(self.count_list[i], dim=1))

        activations = self.encoder.forward(x)
        px_z, stats = self.decoder.forward(activations, get_latents=True,
                                           extended_features=extended_features)
        ndims = np.prod(x.shape[1:])
        if self.gaussian_nll:
            if self.gaussian_nll:
                px_z = self.decoder.out_net(px_z)
                distortion_per_pixel = -Normal(px_z,
                                              F.softplus(self.sigma)).log_prob(
                    x).sum(
                    [1, 2, 3]) / ndims
        else:
            distortion_per_pixel = self.decoder.out_net.nll(px_z,
                                                            x.permute(0, 2, 3,
                                                                      1))
        log_j_z = 0.
        for statdict in stats:
            if 'z_loss' in statdict.keys():
                log_j_z += statdict['z_loss']
        rate_per_pixel = log_j_z / ndims
        loss = (distortion_per_pixel - rate_per_pixel).mean()
        return loss

    def forward_get_latents(self, x):
        extended_features = []
        for i, feature_block in enumerate(self.feature_blocks):
            core = feature_block(x)
            extended_features.extend(core.chunk(self.count_list[i], dim=1))
        activations = self.encoder.forward(x)
        _, stats = self.decoder.forward(activations, get_latents=True,
                                        extended_features=extended_features)
        return stats

    def forward_uncond_samples(self, n_batch, t=None):
        px_z = self.decoder.forward_uncond(n_batch, t=t)
        if self.gaussian_nll:
            return self.decoder.out_net(px_z)
        else:
            return self.decoder.out_net.sample(px_z)

    def forward_samples_set_latents(self, n_batch, latents, t=None):
        px_z = self.decoder.forward_manual_latents(n_batch, latents, t=t)
        if self.gaussian_nll:
            return self.decoder.out_net(px_z)
        else:
            return self.decoder.out_net.sample(px_z)