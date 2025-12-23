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

class ScoreNetTrainBaseAnneal(ModelBase):
    def __init__(
        self,
        score_net,
        pre_score_model=None,
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

    def get_device(self):
        return next(self.score_net.parameters()).device

    def forward(self, x):
        return self.score_net(x)

    def _compute_loss(self, batch):
        device = self.device
        x = batch[0].to(device)

        sigma = np.random.choice(np.array(self.score_sigma), size=(x.shape[0],1), replace=True)
        sigma = torch.from_numpy(sigma).to(device).float()

        if self.pre_score_model:
            pre_scorenet = self.pre_score_model.score_net
            pre_beta = self.pre_score_model.score_beta
        else:
            pre_scorenet = None
            pre_beta = None

        return dsm_score_estimation_mixture(scorenet=self.score_net,
                                            samples=x,
                                            sigma=sigma,
                                            pre_scorenet=pre_scorenet,
                                            beta=self.score_beta,
                                            pre_beta=pre_beta,
                                            alpha=self.config.score_alpha)

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        if isinstance(self.score_net, EMA):
            self.score_net.update_ema()
