import torch
from torch import nn, Tensor
from abc import abstractmethod


class AutoEncoder(nn.Module):

    def __init__(self):
        super().__init__()

    def encode(self, x: Tensor):
        raise NotImplementedError

    def decode(self, z: Tensor):
        raise NotImplementedError

    def sample(self, n_samples: int):
        raise NotImplementedError

    @abstractmethod
    def forward(self, x: Tensor):
        pass

    @abstractmethod
    def loss_function(self, x: Tensor):
        pass
