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
        embed_net,
        *args,
        **kwargs,
    ):
                
        super().__init__(*args, **kwargs)
        self.embed_net = embed_net
        
    def get_device(self):
        return next(self.embed_net.parameters()).device
    
    def G(self, x):
        pass
    
    def forward(self, x):
        return self.G(x)
    
    def _compute_loss(self, batch):
        return torch.tensor(0.0, device=self.device, requires_grad=True)

