from torch import nn
from torch.nn.utils import weight_norm


class MLP(nn.Module):
    '''
      Multilayer Perceptron.
    '''
    def __init__(self, input_size, latent_dim):
        super().__init__()
        self.layers = nn.Sequential(
          nn.Flatten(),
          weight_norm(nn.Linear(input_size, latent_dim)),)
    '''nn.ReLU(),
      weight_norm(nn.Linear(256, 256)),
      nn.ReLU(),
      weight_norm(nn.Linear(256, latent_dim))
    )'''
    '''def __init__(self, input_size, latent_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            weight_norm(nn.Linear(input_size, 256)),
            nn.Tanh(),
            weight_norm(nn.Linear(256, latent_dim)),
        )'''


    def forward(self, x):
      '''Forward pass'''
      return self.layers(x)