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

class EnergyNetTrainBase(ModelBase):
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
        self.energy_sigma = [0.0] + self.score_sigma
        #self._init_weights()
        
    def get_device(self):
        return next(self.energy_net.parameters()).device
    
    def forward(self, x):
        return self.energy_net(x)
    
    def _init_weights(self):
        self.energy_net.load_state_dict(self.score_net.state_dict())

    # energy is -energy_net(x) in this case
    def _energy_grad(self, x):
        def F(x):
            x = x.unsqueeze(0)
            return (self.energy_net(x) @ x.T).squeeze()
        return vmap(jacrev(F))(x)

    def get_energy(self, x):
        return -torch.linalg.vecdot(self.energy_net(x), x)

    def _compute_loss(self, batch):
        device = self.device
        x = batch[0].to(device)

        sigma = np.random.choice(np.array(self.energy_sigma), size=(x.shape[0],1), replace=True)
        sigma = torch.from_numpy(sigma).to(device).float()
        x_hat = x + torch.randn_like(x) * sigma
        dF_x = self._energy_grad(x_hat)
        score = self.score_model(x_hat).detach()
        return 0.5 * (dF_x - score).flatten(1).square().sum(-1).mean()

