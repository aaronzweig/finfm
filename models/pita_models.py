
from typing import Optional

import torch
from torch import nn



import os
import torch
import wandb
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import numpy as np
from torch.optim import AdamW
from torchmetrics.functional import mean_squared_error

from losses.losses import *
from models.base_model import *
from models.ema import *




class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py

        t = t.squeeze(-1)
        
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb







class ScoreNetTrainPita(ModelBase):
    def __init__(
        self,
        score_net,
        pre_score_model,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.score_net = score_net
        self.num_sigmas = self.config.num_sigmas
        self.sigma_min = self.config.sigma_min
        self.sigma_max = self.config.sigma_max
        self.score_sigma = np.exp(np.linspace(np.log(self.sigma_min), np.log(self.sigma_max), self.num_sigmas)).tolist()
        self.score_beta = None
        self.pre_score_model = pre_score_model

        self.sigma_map = TimestepEmbedder(self.config.sigma_dim)
        self.cluster_map = TimestepEmbedder(self.config.sigma_dim)

    def get_device(self):
        return next(self.score_net.parameters()).device

    def forward_noise_predictor(self, x, sigma, cluster):
        sigma_emb = self.sigma_map(torch.log(sigma))
        cluster_emb = self.cluster_map(cluster)
        return self.score_net(torch.cat([x, sigma_emb, cluster_emb], axis = -1))

    def forward(self, x, sigma=None, cluster=None):
        if sigma is None:
            sigma = min(self.score_sigma) * torch.ones(x.shape[0],1,device=x.device)
        if cluster is None:
            cluster = torch.zeros(x.shape[0],1,device=x.device)
            
        return -1/sigma * 1/self.score_beta * self.forward_noise_predictor(x, sigma, cluster)

    def _compute_loss(self, batch):
        device = self.device
        x, y = batch
        x.to(device)
        y.to(device)

        sigma = np.random.choice(np.array(self.score_sigma), size=(x.shape[0],1), replace=True)
        sigma = torch.from_numpy(sigma).to(device).float()

        noise = torch.randn_like(x)
        x_hat = x + noise * sigma
        
        noise_pred = self.forward_noise_predictor(x_hat, sigma, y)
        loss = ((noise_pred - noise) ** 2).sum(dim=-1)
    
        return loss.mean()


    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        if isinstance(self.score_net, EMA):
            self.score_net.update_ema()


from torch.func import jvp, vmap, jacrev

class EnergyNetTrainPita(ModelBase):
    def __init__(
        self,
        score_model,
        energy_net,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.score_model = score_model
        self.energy_net = energy_net
        self.num_sigmas = self.config.num_sigmas
        self.sigma_min = self.config.sigma_min
        self.sigma_max = self.config.sigma_max
        self.score_sigma = np.exp(np.linspace(np.log(self.sigma_min), np.log(self.sigma_max), self.num_sigmas)).tolist()
        self.energy_sigma = self.score_sigma

        self.sigma_map = TimestepEmbedder(self.config.sigma_dim)
        self.cluster_map = TimestepEmbedder(self.config.sigma_dim)
        
    def get_device(self):
        return next(self.energy_net.parameters()).device        
    
    def forward_energy(self, x, sigma, cluster):

        def f_theta(x, sigma, cluster):
            sigma_emb = self.sigma_map(sigma)
            cluster_emb = self.cluster_map(cluster)
            h_theta = self.energy_net(torch.cat([x, sigma_emb, cluster_emb], dim=-1))
            return torch.sum(h_theta * x, dim=1)

        U_theta = f_theta(x, sigma, cluster)
        return -U_theta

    def get_energy(self, x, cluster=None):
        if cluster is None:
            cluster = torch.zeros(x.shape[0],1).to(x.device)
        sigma = torch.ones(x.shape[0],1).to(x.device) * min(self.score_sigma) ### FROZEN CHOICE OF INTERPOLATION?
        return self.forward_energy(x, sigma, cluster)
    
    def forward(self, x, sigma, cluster):
        U = self.forward_energy(x, sigma, cluster)
        nabla_U = torch.autograd.grad(U.sum(), x, create_graph=True)[0]
        return nabla_U

    def _compute_loss(self, batch):
        device = self.device
        x, y = batch
        x.to(device)
        y.to(device)


        sigma = np.random.choice(np.array(self.energy_sigma), size=(x.shape[0],1), replace=True)
        sigma = torch.from_numpy(sigma).to(device).float()

        x.requires_grad = True
        sigma.requires_grad = True
        
        x_hat = x + torch.randn_like(x) * sigma
        dF_x = self.forward(x_hat, sigma, y)
        score = self.score_model(x_hat, sigma, y).detach()
        return 0.5 * (dF_x + score).flatten(1).square().sum(-1).mean()
