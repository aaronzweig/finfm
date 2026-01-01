import torch.nn.functional as F
import torch.nn as nn
import torch.nn.init as init
import torch
import numpy as np
from typing import List, Optional
import math

class SinNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_freq: int,
        activation: str = "selu",
        layer_norm: bool = False,
        hidden_dims: List[int] = None,
        rescale: float = 1.0,
    ):
        super().__init__()
        
        self.num_freq = num_freq
        self.model = SimpleDenseNet(input_dim=input_dim + 2 * num_freq,
                                    output_dim=output_dim,
                                    activation=activation,
                                    layer_norm=layer_norm,
                                    hidden_dims=hidden_dims)
        self.rescale = rescale

    def sinusoidal_time_encoding(self, t, max_period=10000):
        device = next(self.parameters()).device

        if t.ndim == 1:
            t = t.unsqueeze(1)
        
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=self.num_freq, dtype=torch.float32)
            / self.num_freq
        ).unsqueeze(0).to(device)
        
        scaled_t = t * freqs
        return torch.cat([torch.cos(scaled_t), torch.sin(scaled_t)], dim=-1)
    
    def forward(self, x, t):
        t = self.sinusoidal_time_encoding(t)
        return self.rescale * self.model(torch.cat([x, t], dim = 1))

    
#https://github.com/kksniak/metric-flow-matching/blob/main/mfm/networks/mlp_base.py

class swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

ACTIVATION_MAP = {
    "relu": nn.ReLU,
    "sigmoid": nn.Sigmoid,
    "tanh": nn.Tanh,
    "selu": nn.SELU,
    "elu": nn.ELU,
    "lrelu": nn.LeakyReLU,
    "softplus": nn.Softplus,
    "silu": nn.SiLU,
    "swish": swish,
}


class ResidualBlock(nn.Module):
    def __init__(self, dim, activation):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            activation(),
            nn.Linear(dim, dim),
        )

    def forward(self, x):
        return x + self.net(x)

class SimpleEmbedNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        sample_rescale,
        activation: str = "selu",
        layer_norm: bool = False,
        hidden_dims: List[int] = None,
        rescale: float = 1.0,
        skip: bool = False,
    ):
        super().__init__()
        self.model = SimpleDenseNet(input_dim=input_dim,
                               output_dim=output_dim,
                               activation=activation,
                               layer_norm=layer_norm,
                               hidden_dims=hidden_dims)
        
        self.rescale = rescale
        self.skip = skip

        if skip: 
            self.D = torch.diag(sample_rescale.squeeze(0)).float()
            if input_dim <= output_dim:
                self.Q = torch.eye(output_dim, input_dim)
            else: 
                #approximately norm-preserving projection
                self.Q = 1/np.sqrt(output_dim) * torch.randn(output_dim, input_dim) 

    def forward(self, x):
        skip_embed = 0
        if self.skip:
            device = next(self.parameters()).device
            skip_embed = x @ self.D.to(device) @ self.Q.T.to(device)
        return skip_embed + self.rescale * self.model(x)
        

class SimpleScoreNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        activation: str = "selu",
        hidden_dim: int = 128,
        num_layers: int = 3,
    ):
        super().__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.LayerNorm(hidden_dim))
        layers.append(ACTIVATION_MAP[activation]())
        for i in range(1, num_layers - 2):
            layers.append(ResidualBlock(hidden_dim, ACTIVATION_MAP[activation]))
        layers.append(nn.LayerNorm(hidden_dim))
        layers.append(ACTIVATION_MAP[activation]())
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class SimpleDenseNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        activation: str = "selu",
        layer_norm: bool = False,
        hidden_dims: List[int] = None,
    ):
        super().__init__()
        dims = [input_dim, *hidden_dims, output_dim]
        layers = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if layer_norm:
                layers.append(nn.LayerNorm(dims[i + 1]))
            layers.append(ACTIVATION_MAP[activation]())
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
