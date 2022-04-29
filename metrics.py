import torch
import numpy as np
import torchvision
from scipy import linalg
from torch import nn
from torch.hub import load_state_dict_from_url
import torch.nn.functional as F
import util, datasets


def calculate_fid(model, dataset, device, n_samples=1024, batch_size=32):

    test_loader = datasets.get_test_dataloader(dataset, batch_size)

    activations_test = np.empty((n_samples, InceptionV3.DEFAULT_DIMS))

    incept = InceptionV3().to(device)
    incept.eval()
    start_idx = 0
    for batch, _ in test_loader:
        if batch.shape[1] == 1:
            # HACK: Inception expects three channels so we tile
            batch = batch.repeat((1, 3, 1, 1))

        with torch.no_grad():
            batch_activations = incept(batch)[0].squeeze(3).squeeze(2).cpu().numpy()

        activations_test[start_idx:start_idx + batch_size, :] = batch_activations
        start_idx = start_idx + batch_size
        if start_idx >= n_samples:
            break

    train_loader = \
    datasets.get_train_val_dataloaders(dataset, batch_size, p_validation=0, return_img_dim=False, return_alpha=False)[0]

    activations_train = np.empty((n_samples, InceptionV3.DEFAULT_DIMS))

    start_idx = 0
    for batch, _ in train_loader:
        if batch.shape[1] == 1:
            # HACK: Inception expects three channels so we tile
            batch = batch.repeat((1, 3, 1, 1))

        with torch.no_grad():
            batch_activations = incept(batch)[0].squeeze(3).squeeze(2).cpu().numpy()

        activations_train[start_idx:start_idx + batch_size, :] = batch_activations
        start_idx = start_idx + batch_size
        if start_idx >= n_samples:
            break

    activations_samples = np.empty((n_samples, InceptionV3.DEFAULT_DIMS))
    n_filled = 0
    min_max_samples = np.empty((n_samples, 2))
    while n_filled < n_samples:
        samples = model.sample(batch_size, temperature=1)
        min_max_samples[n_filled:n_filled + batch_size, 0] = torch.min(samples.view(batch_size, -1).cpu().detach(),
                                                                       dim=1).values.numpy()
        min_max_samples[n_filled:n_filled + batch_size, 1] = torch.max(samples.view(batch_size, -1).cpu().detach(),
                                                                       dim=1).values.numpy()
        if samples.shape[1] == 1:
            # HACK: Inception expects three channels so we tile
            samples = samples.repeat((1, 3, 1, 1))

            with torch.no_grad():
                batch_activations = incept(samples)[0].squeeze(3).squeeze(2).cpu().numpy()
            activations_samples[n_filled:n_filled + batch_size] = batch_activations
            n_filled += batch_size

    train_mu, train_cov = get_statistics_numpy(activations_train)
    test_mu, test_cov = get_statistics_numpy(activations_test)
    samples_mu, samples_cov = get_statistics_numpy(activations_samples)

    fid_test_sample = calculate_frechet_distance(samples_mu, samples_cov, test_mu, test_cov)
    fid_test_train = calculate_frechet_distance(train_mu, train_cov, test_mu, test_cov)
    fid_train_sample = calculate_frechet_distance(train_mu, train_cov, samples_mu, samples_cov)
    print(f'FID between test and train set: {fid_test_train}')
    print(f'FID between test set and generated samples: {fid_test_sample}')
    print(f'FID between train set and generated samples: {fid_train_sample}')
    return fid_test_train, fid_test_sample, fid_train_sample


class SampleLoader:
    def __init__(self, batch_size, num_total_samples, density):
        self.batch_size = batch_size
        self.num_total_samples = num_total_samples
        self.density = density

    def __iter__(self):
        self.num_remaining_samples = self.num_total_samples
        return self

    def __next__(self):
        if self.batch_size < self.num_remaining_samples:
            num_samples = self.batch_size
        elif self.num_remaining_samples > 0:
            num_samples = self.num_remaining_samples
        else:
            raise StopIteration

        samples = self.density.sample(num_samples)
        self.num_remaining_samples -= num_samples

        return samples, None # HACK: return tuple to conform to SupervisedDataset signature



def get_fid_function(config, train_loader):
    train_dataset = train_loader.dataset.x

    if config["dataset"] in ["mnist", "fashion-mnist", "svhn", "cifar10"]:
        get_data_fn = get_inception_activations
    else:
        get_data_fn = get_data_from_loader

    train_data_numpy = get_data_fn(
        dataloader=train_loader,
        length=train_dataset.shape[0],
        device=train_dataset.device
    )
    train_mu, train_cov = get_statistics_numpy(train_data_numpy)

    def fid_function(density):
        sample_loader = SampleLoader(
            batch_size=config["test_batch_size"],
            num_total_samples=config["num_fid_samples"],
            density=density
        )
        sample_data = get_data_fn(
            dataloader=sample_loader,
            length=config["num_fid_samples"],
            device=train_dataset.device
        )
        sample_mu, sample_cov = get_statistics_numpy(sample_data)

        return calculate_frechet_distance(
            mu1=sample_mu,
            sigma1=sample_cov,
            mu2=train_mu,
            sigma2=train_cov
        )

    return fid_function


#### NOTE: Below adapted from
# https://github.com/mseitzer/pytorch-fid/blob/master/src/pytorch_fid/fid_score.py

def get_statistics_numpy(numpy_data):
    mu = np.mean(numpy_data, axis=0)
    cov = np.cov(numpy_data, rowvar=False)
    return mu, cov


def get_data_from_loader(dataloader, length, device):
    start_idx = 0

    for batch, _ in dataloader:
        if start_idx == 0:
            data = np.empty((length, *batch.shape[1:]))

        data[start_idx:start_idx+batch.shape[0]] = batch.detach().cpu().numpy()
        start_idx = start_idx + batch.shape[0]

    return data


def get_inception_activations(dataloader, length, device):
    # NOTE: Store in numpy array for higher precision
    activations = np.empty((length, InceptionV3.DEFAULT_DIMS))
    start_idx = 0

    model = InceptionV3().to(device)
    model.eval()

    for batch, _ in dataloader:
        if batch.shape[1] == 1:
            # HACK: Inception expects three channels so we tile
            batch = batch.repeat((1,3,1,1))

        with torch.no_grad():
            batch_activations = model(batch)[0].squeeze(3).squeeze(2).cpu().numpy()

        activations[start_idx:start_idx+batch.shape[0]] = batch_activations
        start_idx = start_idx + batch.shape[0]

    return activations


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1)
            + np.trace(sigma2) - 2 * tr_covmean)


#### NOTE: Below taken from
# https://github.com/mseitzer/pytorch-fid/blob/master/src/pytorch_fid/inception.py

# Inception weights ported to Pytorch from
# http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
FID_WEIGHTS_URL = 'https://github.com/mseitzer/pytorch-fid/releases/download/fid_weights/pt_inception-2015-12-05-6726825d.pth'  # noqa: E501


class InceptionV3(nn.Module):
    """Pretrained InceptionV3 network returning feature maps"""
    DEFAULT_DIMS = 2048

    # Index of default block of inception to return,
    # corresponds to output of final average pooling
    DEFAULT_BLOCK_INDEX = 3

    # Maps feature dimensionality to their output blocks indices
    BLOCK_INDEX_BY_DIM = {
        64: 0,   # First max pooling features
        192: 1,  # Second max pooling featurs
        768: 2,  # Pre-aux classifier features
        2048: 3  # Final average pooling features
    }

    def __init__(self,
                 output_blocks=(DEFAULT_BLOCK_INDEX,),
                 resize_input=True,
                 normalize_input=True,
                 requires_grad=False,
                 use_fid_inception=True):
        """Build pretrained InceptionV3
        Parameters
        ----------
        output_blocks : list of int
            Indices of blocks to return features of. Possible values are:
                - 0: corresponds to output of first max pooling
                - 1: corresponds to output of second max pooling
                - 2: corresponds to output which is fed to aux classifier
                - 3: corresponds to output of final average pooling
        resize_input : bool
            If true, bilinearly resizes input to width and height 299 before
            feeding input to model. As the network without fully connected
            layers is fully convolutional, it should be able to handle inputs
            of arbitrary size, so resizing might not be strictly needed
        normalize_input : bool
            If true, scales the input from range (0, 1) to the range the
            pretrained Inception network expects, namely (-1, 1)
        requires_grad : bool
            If true, parameters of the model require gradients. Possibly useful
            for finetuning the network
        use_fid_inception : bool
            If true, uses the pretrained Inception model used in Tensorflow's
            FID implementation. If false, uses the pretrained Inception model
            available in torchvision. The FID Inception model has different
            weights and a slightly different structure from torchvision's
            Inception model. If you want to compute FID scores, you are
            strongly advised to set this parameter to true to get comparable
            results.
        """
        super(InceptionV3, self).__init__()

        self.resize_input = resize_input
        self.normalize_input = normalize_input
        self.output_blocks = sorted(output_blocks)
        self.last_needed_block = max(output_blocks)

        assert self.last_needed_block <= 3, \
            'Last possible output block index is 3'

        self.blocks = nn.ModuleList()

        if use_fid_inception:
            inception = fid_inception_v3()
        else:
            inception = _inception_v3(pretrained=True)

        # Block 0: input to maxpool1
        block0 = [
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2)
        ]
        self.blocks.append(nn.Sequential(*block0))

        # Block 1: maxpool1 to maxpool2
        if self.last_needed_block >= 1:
            block1 = [
                inception.Conv2d_3b_1x1,
                inception.Conv2d_4a_3x3,
                nn.MaxPool2d(kernel_size=3, stride=2)
            ]
            self.blocks.append(nn.Sequential(*block1))

        # Block 2: maxpool2 to aux classifier
        if self.last_needed_block >= 2:
            block2 = [
                inception.Mixed_5b,
                inception.Mixed_5c,
                inception.Mixed_5d,
                inception.Mixed_6a,
                inception.Mixed_6b,
                inception.Mixed_6c,
                inception.Mixed_6d,
                inception.Mixed_6e,
            ]
            self.blocks.append(nn.Sequential(*block2))

        # Block 3: aux classifier to final avgpool
        if self.last_needed_block >= 3:
            block3 = [
                inception.Mixed_7a,
                inception.Mixed_7b,
                inception.Mixed_7c,
                nn.AdaptiveAvgPool2d(output_size=(1, 1))
            ]
            self.blocks.append(nn.Sequential(*block3))

        for param in self.parameters():
            param.requires_grad = requires_grad

    def forward(self, inp):
        """Get Inception feature maps
        Parameters
        ----------
        inp : torch.autograd.Variable
            Input tensor of shape Bx3xHxW. Values are expected to be in
            range (0, 1)
        Returns
        -------
        List of torch.autograd.Variable, corresponding to the selected output
        block, sorted ascending by index
        """
        outp = []
        x = inp

        if self.resize_input:
            x = F.interpolate(x,
                              size=(299, 299),
                              mode='bilinear',
                              align_corners=False)

        if self.normalize_input:
            x = 2 * x - 1  # Scale from range (0, 1) to range (-1, 1)

        for idx, block in enumerate(self.blocks):
            x = block(x)
            if idx in self.output_blocks:
                outp.append(x)

            if idx == self.last_needed_block:
                break

        return outp


def _inception_v3(*args, **kwargs):
    """Wraps `torchvision.models.inception_v3`
    Skips default weight inititialization if supported by torchvision version.
    See https://github.com/mseitzer/pytorch-fid/issues/28.
    """
    try:
        version = tuple(map(int, torchvision.__version__.split('.')[:2]))
    except ValueError:
        # Just a caution against weird version strings
        version = (0,)

    if version >= (0, 6):
        kwargs['init_weights'] = False

    return torchvision.models.inception_v3(*args, **kwargs)


def fid_inception_v3():
    """Build pretrained Inception model for FID computation
    The Inception model for FID computation uses a different set of weights
    and has a slightly different structure than torchvision's Inception.
    This method first constructs torchvision's Inception and then patches the
    necessary parts that are different in the FID Inception model.
    """
    inception = _inception_v3(num_classes=1008,
                              aux_logits=False,
                              pretrained=False)
    inception.Mixed_5b = FIDInceptionA(192, pool_features=32)
    inception.Mixed_5c = FIDInceptionA(256, pool_features=64)
    inception.Mixed_5d = FIDInceptionA(288, pool_features=64)
    inception.Mixed_6b = FIDInceptionC(768, channels_7x7=128)
    inception.Mixed_6c = FIDInceptionC(768, channels_7x7=160)
    inception.Mixed_6d = FIDInceptionC(768, channels_7x7=160)
    inception.Mixed_6e = FIDInceptionC(768, channels_7x7=192)
    inception.Mixed_7b = FIDInceptionE_1(1280)
    inception.Mixed_7c = FIDInceptionE_2(2048)

    state_dict = load_state_dict_from_url(FID_WEIGHTS_URL, progress=True)
    inception.load_state_dict(state_dict)
    return inception


class FIDInceptionA(torchvision.models.inception.InceptionA):
    """InceptionA block patched for FID computation"""
    def __init__(self, in_channels, pool_features):
        super(FIDInceptionA, self).__init__(in_channels, pool_features)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        # Patch: Tensorflow's average pool does not use the padded zero's in
        # its average calculation
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1,
                                   count_include_pad=False)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class FIDInceptionC(torchvision.models.inception.InceptionC):
    """InceptionC block patched for FID computation"""
    def __init__(self, in_channels, channels_7x7):
        super(FIDInceptionC, self).__init__(in_channels, channels_7x7)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)

        branch7x7dbl = self.branch7x7dbl_1(x)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)

        # Patch: Tensorflow's average pool does not use the padded zero's in
        # its average calculation
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1,
                                   count_include_pad=False)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch7x7, branch7x7dbl, branch_pool]
        return torch.cat(outputs, 1)


class FIDInceptionE_1(torchvision.models.inception.InceptionE):
    """First InceptionE block patched for FID computation"""
    def __init__(self, in_channels):
        super(FIDInceptionE_1, self).__init__(in_channels)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [
            self.branch3x3_2a(branch3x3),
            self.branch3x3_2b(branch3x3),
        ]
        branch3x3 = torch.cat(branch3x3, 1)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [
            self.branch3x3dbl_3a(branch3x3dbl),
            self.branch3x3dbl_3b(branch3x3dbl),
        ]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)

        # Patch: Tensorflow's average pool does not use the padded zero's in
        # its average calculation
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1,
                                   count_include_pad=False)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class FIDInceptionE_2(torchvision.models.inception.InceptionE):
    """Second InceptionE block patched for FID computation"""
    def __init__(self, in_channels):
        super(FIDInceptionE_2, self).__init__(in_channels)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [
            self.branch3x3_2a(branch3x3),
            self.branch3x3_2b(branch3x3),
        ]
        branch3x3 = torch.cat(branch3x3, 1)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [
            self.branch3x3dbl_3a(branch3x3dbl),
            self.branch3x3dbl_3b(branch3x3dbl),
        ]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)

        # Patch: The FID Inception model uses max pooling instead of average
        # pooling. This is likely an error in this specific Inception
        # implementation, as other Inception models use average pooling here
        # (which matches the description in the paper).
        branch_pool = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)