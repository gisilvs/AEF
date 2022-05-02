import torch

from bijectors.actnorm import ActNorm
from bijectors.realnvp import get_realnvp_bijector
from bijectors.masked_autoregressive_transform import get_masked_autoregressive_transform
from flows.maf import MaskedAutoregressiveFlow
from bijectors.glow import InvertibleConv1x1



def test_flat_bijector(bijector):
    input_tensor = torch.randn([23, 45])
    z, ildj = bijector.inverse(input_tensor)  # ildj --> inverse log determinant jacobian
    x, fldj = bijector.forward(z)  # fldj --> forward log determinant jacobian
    if not torch.allclose(input_tensor, x, atol=1e-6):
        return False
    if not torch.allclose(ildj, -fldj):
        return False

    return True


def test_image_bijector(bijector):
    input_tensor = torch.randn([23,3,32,32])
    z, ildj = bijector.inverse(input_tensor)  # ildj --> inverse log determinant jacobian
    x, fldj = bijector.forward(z)  # fldj --> forward log determinant jacobian
    if not torch.allclose(input_tensor, x, atol=1e-6):
        return False
    if not torch.allclose(ildj, -fldj):
        return False
    return True


def main():
    print('Tests on 2D data')
    flat_bijectors = {'RealNVP': get_realnvp_bijector(45, 256, 4, 2),
                      'ActNorm': ActNorm(1),
                      'MaskedAutoregressiveTransform': get_masked_autoregressive_transform(45, 256, 8, 2, act_norm_between_layers=True),
                      'MaskedAutoregressiveFlow': MaskedAutoregressiveFlow(45,256, 8, 2, act_norm_between_layers=True)}
    for name, bijector in flat_bijectors.items():
        if test_flat_bijector(bijector):
            print(f'{name} test PASSED')
        else:
            print(f'{name} test FAILED')

    print('Tests on 4D data')
    image_bijectors = {
        'ActNorm': ActNorm(3),
        'OneByOneConv': InvertibleConv1x1(3, None)
        #'Glow': SimpleGlow(8,3,256)
    }
    for name, bijector in image_bijectors.items():
        if test_image_bijector(bijector):
            print(f'{name} test PASSED')
        else:
            print(f'{name} test FAILED')


if __name__ == "__main__":
    main()
