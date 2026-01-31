import os
import torch
import wandb
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from torch.optim import AdamW
from torchmetrics.functional import mean_squared_error
from torchdyn.core import NeuralODE
from torchvision import transforms
import torch.nn.functional as F

from torchcfm.models import MLP
from torchcfm.utils import plot_trajectories, torch_wrapper

from models.base_model import ModelBase

class ClassifierNetTrainBase(ModelBase):
    def __init__(
        self,
        classifier_net,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.classifier_net = classifier_net
        self.dummy_prior = self.config.dummy_prior
        self.dummy_kl_weight = self.config.dummy_kl_weight

    #Expects input from raw data
    def classify(self, x): 
        x = self.normalize(x)
        return self.classifier_net(x)

    def forward(self, x, normalize=False):
        if normalize:
            x = self.normalize(x)
        return self.classifier_net(x)
    
    def _prepare_batch(self, batch):

        device = self.get_device()

        x, y = batch
        x.to(device)
        y.to(device)
        x = self.normalize(x)

        return x, y

    def _compute_loss(self, batch):
        x, y = self._prepare_batch(batch)
        logits = self.classifier_net(x)
        ce_loss = F.cross_entropy(logits, y)

        kl_loss = torch.tensor(0.0, device=logits.device)
        if self.dummy_kl_weight > 0:
            num_logits = logits.shape[-1]
            prior = torch.full((num_logits,), 1.0 / num_logits, device=logits.device)
            if num_logits > 1:
                base_prob = (1.0 - self.dummy_prior) / (num_logits - 1)
                prior.fill_(base_prob)
                prior[-1] = self.dummy_prior
            log_probs = F.log_softmax(logits, dim=-1)
            kl_loss = F.kl_div(log_probs, prior, reduction='batchmean', log_target=False)

        loss = ce_loss + self.dummy_kl_weight * kl_loss
        if self.dummy_kl_weight > 0:
            self.log("train_ce", ce_loss.detach())
            self.log("train_kl", kl_loss.detach())
        return loss
        
    def training_step(self, batch, batch_idx):
        loss = self._compute_loss(batch)

        self.log("train_loss_classifier", loss)
        return loss