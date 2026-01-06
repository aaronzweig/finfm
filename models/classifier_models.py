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

    def forward(self, x):
        return self.classifier_net(x)
    
    def _prepare_batch(self, batch):

        device = self.get_device()

        x, y = batch
        x.to(device)
        y.to(device)

        return x, y

    def _compute_loss(self, batch):
        x, y = self._prepare_batch(batch)
        logits = self.classifier_net(x)
        loss = F.cross_entropy(logits, y)
        return loss
    #TODO: add outlier class
        