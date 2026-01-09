import os
import torch
import wandb
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from torch.optim import AdamW
from torchmetrics.functional import mean_squared_error
from torchdyn.core import NeuralODE
from torchvision import transforms

from torchcfm.models import MLP
from torchcfm.utils import plot_trajectories, torch_wrapper

class ModelBase(pl.LightningModule):
    def __init__(
        self,
        config,
        *args, 
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.config = config
        self.lr = config.lr
        self.warmup_steps = config.warmup_steps

    def get_device(self):
        return self.device

    def forward(self, x, t):
        pass

    def _compute_loss(self, batch):
        pass

    def training_step(self, batch, batch_idx):
        loss = self._compute_loss(batch)

        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.lr)
    
        def lr_lambda(current_step):
            warmup_steps = self.warmup_steps
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return 1.0  # Or add decay logic here
    
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda),
            'interval': 'step',  # update every optimizer step
            'frequency': 1,
        }
    
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
