import os
import torch
import torch.nn.functional as F
import wandb
import numpy as np
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from torch.optim import AdamW
from torch.func import jvp, vmap, jacrev
from torchcfm.conditional_flow_matching import pad_t_like_x, ConditionalFlowMatcher
from torchmetrics.functional import mean_squared_error

from .base_model import *
from utils.frozen import *

class MetricNetTrainBase(ModelBase):
    def __init__(
        self,
        metric_net,
        energy_model,
        *args,
        **kwargs,
    ):
        
        freeze_params(energy_model)
        
        super().__init__(*args, **kwargs)
        self.metric_net = metric_net
        self.energy_model = energy_model
        self.num_sigmas = self.config.num_sigmas
        self.sigma_min = self.config.sigma_min
        self.sigma_max = self.config.sigma_max
        self.score_sigma = np.exp(np.linspace(np.log(self.sigma_min), np.log(self.sigma_max), self.num_sigmas)).tolist()
        self.energy_sigma = [0.0] + self.score_sigma

        self.metric_sigma = self.config.metric_sigma

        # cutoffs for energy
        self.low_quantile = None
        self.high_quantile = None
        self.gamma = self.config.gamma
        
    def get_device(self):
        return next(self.metric_net.parameters()).device
    
    def G(self, x):
        E = self.energy_model.get_energy(x)
        E = torch.clamp(E - self.low_quantile, min=0)
        E = self.gamma + E * self.config.metric_scale

        return E

    # def forward(self, x):
    #     return self.metric_net(x).squeeze(-1)
    
    # def _compute_loss(self, batch):
    #     device = self.device
    #     x = batch[0].to(device)

    #     sigma = np.random.choice(np.array(self.energy_sigma), size=(x.shape[0],1), replace=True)
    #     sigma = torch.from_numpy(sigma).to(device).float()
    #     x = x + torch.randn_like(x) * sigma
    #     x_hat = x + torch.randn_like(x) * self.metric_sigma

    #     return (self(x) - self.G(x_hat)).square().mean()

    ### Mute metric learning
    
    def forward(self, x):
        return self.G(x)
    
    def _compute_loss(self, batch):
        return torch.tensor(0.0, device=self.device, requires_grad=True)

