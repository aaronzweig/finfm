import torch
from torch import nn
from torch.nn import functional as F
from typing import List, Optional
import pytorch_lightning as pl
from .modules import *

import wandb
from torch.optim import AdamW

from .base_model import *


#https://github.com/AntixK/PyTorch-VAE/blob/master/models/cvae.py
class ConditionalVAE(ModelBase):

    def __init__(self,
            encoder,
            decoder,
            **kwargs,
        ):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder        
        self.beta = self.config.beta
        self.latent_dim = self.config.latent_dim

    def get_device(self):
        return next(self.encoder.parameters()).device

    def encode(self, x, c, t):
        x_c = torch.cat([x, c], dim=-1)
        z = self.encoder(x_c, t)
        mu, log_var = torch.chunk(z, 2, dim=-1)
        return [mu, log_var]

    def decode(self, z, c, t):
        z_c = torch.cat([z, c], dim=-1)
        z = self.decoder(z_c, t)
        return z

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, x, c, t):
        mu, log_var = self.encode(x, c, t)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decode(z, c, t)
        return  x_recon, mu, log_var

    def loss_function(self, x, c, t):
        x_recon, mu, log_var = self(x, c, t)

        recon_loss = F.mse_loss(x_recon, x)

        kl_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        loss = recon_loss + self.beta * kl_loss
        return loss

    #TODO: the vae can handle mixed dataloader better than flow matching
    def _compute_loss(self, batch):

        device = self.get_device()

        x0, x1, t0, t1, c = self._prepare_batch(batch)

        t0 = torch.tensor(t0).unsqueeze(0).repeat(x0.shape[0]).to(device)
        t1 = torch.tensor(t1).unsqueeze(0).repeat(x1.shape[0]).to(device)
        
        loss = self.loss_function(x0, c, t0)
        loss += self.loss_function(x1, c, t1)

        return loss


    def sample(self, x, c, t0, t1, num_samples):

        del t0
        del x

        device = self.get_device()

        t1 = torch.tensor(t1).unsqueeze(0).repeat(num_samples).to(device)
        z = torch.randn(num_samples, self.latent_dim).to(device)
        samples = self.decode(z, c, t1)
        return samples

