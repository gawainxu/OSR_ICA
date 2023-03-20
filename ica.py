"""
The code used to train nueral networks to 
estimate component correlation in features

https://github.com/pbrakel/anica
"""

import torch
import torch.nn as nn
from torchsummary import summary

import numpy as np

class ica_mlp(nn.Module):
    def __init__(self, dims, stddev=1., bias_value=0.0):
        super().__init__()
        self.dims = dims
        self.layers = []
        previous_dim = dims[0]

        for i, dim in enumerate(dims[1:]):
            layer = nn.Linear(previous_dim, dim)
            nn.init.trunc_normal_(layer.weight, std=stddev/np.sqrt(previous_dim))

            if i < len(dims) - 2:
                nn.init.constant_(layer.bias, val=bias_value)
            else:
                nn.init.constant_(layer.bias, 0.0)

            self.layers.append(layer)
            if i < len(dims) - 2:
                self.layers.append(nn.ReLU())

            previous_dim = dim
        
        self.layers = nn.ModuleList(self.layers)


    def forward(self, x):

        for i, l in enumerate(self.layers):
            x = l(x)

        return x


if __name__ == "__main__":

    dims = [128, 50, 50, 1]
    activations = [nn.ReLU(), nn.ReLU(), None]
    
    mlp = ica_mlp(dims=dims)
    summary(mlp, (128,))