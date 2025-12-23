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
from sklearn.cluster import KMeans


from models.base_model import *

class MetricNetMFM(ModelBase):
    def __init__(
        self,
        metric_net,
        K = 20,
        kappa = 1.0,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.metric_net = metric_net #REDUNDANT, not needed by this module
        self.K = K
        self.clustering_model = KMeans(n_clusters=self.K)
        self.kappa = kappa
        self.W = torch.nn.Parameter(torch.rand(self.K, 1))
        
    def get_device(self):
        return self.device

    def forward(self, x, alpha=1, epsilon=1e-2):
        dist2 = torch.cdist(x, self.C) ** 2
        self.phi_x = torch.exp(-0.5 * self.lamda[None, :, :] * dist2[:, :, None])
        h_x = (self.W.to(x.device) * self.phi_x).sum(dim=1)
        
        M_x = 1 / (h_x + epsilon) ** alpha
        return M_x

    def on_before_zero_grad(self, *args, **kwargs):
        self.W.data = torch.clamp(self.W.data, min=0.0001)

    def on_train_start(self):
        with torch.no_grad():

            data_to_fit = []
            for batch, *_ in self.train_dataloader:
                data_to_fit.append(batch.detach().cpu())
            data_to_fit = torch.cat(data_to_fit)

            print("Fitting Clustering model...")
            self.clustering_model.fit(data_to_fit)

            clusters = self.clustering_model.cluster_centers_

            self.C = torch.tensor(clusters, dtype=torch.float32).to(self.device)
            labels = self.clustering_model.labels_
            sigmas = np.zeros((self.K, 1))

            for k in range(self.K):
                points = data_to_fit[labels == k, :]
                variance = ((points - clusters[k]) ** 2).mean(axis=0)
                sigmas[k, :] = np.sqrt(variance.mean())

            self.lamda = torch.tensor(
                0.5 / (self.kappa * sigmas + 1e-8) ** 2, dtype=torch.float32
            ).to(self.device)
    
    def _compute_loss(self, batch):
        device = self.device
        x = batch[0].to(device)
        loss = ((1 - self.forward(x)) ** 2).mean()
        return loss


