import os
import torch
import torch.nn.functional as F
import wandb
import numpy as np
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from torch.optim import AdamW
from torchmetrics.functional import mean_squared_error
from torchdyn.core import NeuralODE
from torchvision import transforms

# from torchcfm.conditional_flow_matching import *
from torchcfm.models import MLP
from torchcfm.utils import plot_trajectories, torch_wrapper

from .base_model import *
from torchdyn.core import NeuralODE

import random


class WrappedVectorField(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, t, x, *args, **kwargs):
        t = t.repeat(x.shape[0])
        return self.model(x, t)

# From https://github.com/kksniak/metric-flow-matching/blob/main/mfm/flow_matchers/flow_net_train.py
class FlowNetTrainBase(ModelBase):
    def __init__(
        self,
        flow_matcher,
        flow_net,
        t_global_min,
        t_global_max,
        sample_rescale,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.flow_matcher = flow_matcher
        self.flow_net = flow_net
        self.t_global_min = t_global_min
        self.t_global_max = t_global_max
        
        self.sample_rescale = sample_rescale
        
    def normalize_time(self, t):
        return (t - self.t_global_min) / (self.t_global_max - self.t_global_min)
    
    def forward(self, x, t):
        return self.flow_net(x, t)

    def _prepare_batch(self, batch):

        device = self.get_device()

        x0, x1, t0, t1 = batch
        x0.to(device)
        x1.to(device)
        t0.to(device)
        t1.to(device)
        
        return x0, x1, t0, t1

    def _compute_loss(self, batch):

        x0, x1, t0, t1 = self._prepare_batch(batch)
        t0 = self.normalize_time(t0)
        t1 = self.normalize_time(t1)

        loss = 0

        for i in range(x0.shape[0]):
        
            t, xt, ut = self.flow_matcher.sample_location_and_conditional_flow(x0[i], x1[i], t0[i], t1[i])    
            vt = self(xt, t)
    
            loss += torch.mean((vt - ut) ** 2)

        return loss / x0.shape[0]

    def sample_traj(self, x, t0, t1, num_samples, steps=2000):

        device = self.get_device()

        x /= self.sample_rescale.to(device)

        t0 = self.normalize_time(t0)
        t1 = self.normalize_time(t1)
    
        node = NeuralODE(WrappedVectorField(self.flow_net), solver="dopri5", sensitivity="adjoint")
        indices = np.random.choice(x.shape[0], num_samples, replace=False)
        indices = torch.from_numpy(indices).to(device)
        with torch.no_grad():
            traj = node.trajectory(
                x[indices].float().to(device),
                t_span=torch.linspace(t0, t1, steps),
            ).cpu()

        return traj * self.sample_rescale.unsqueeze(0).to(device)

    def sample(self, x, t0, t1, num_samples, steps=2000):
        traj = self.sample_traj(x, t0, t1, num_samples, steps)
        return traj[-1]

    def sample_and_weight(self, x, t0, t1, num_samples, steps=2000):
        samples = self.sample(x, t0, t1, num_samples, steps)
        return samples, None


############################################################################################################################
#GROWTH?
############################################################################################################################

# class WrappedVectorFieldGrowth(torch.nn.Module):
#     def __init__(self, flow_model, growth_model, c, w=1.0):
#         super().__init__()
#         self.flow_model = flow_model
#         self.growth_model = growth_model
#         self.c = c
#         self.w = w

#     def forward(self, t, x, *args, **kwargs):
#         x = x[..., :-1]
#         t = t.repeat(x.shape[0])
#         v = self.flow_model(x, self.c, t)
#         h = self.growth_model(x, self.c, t)
#         return torch.cat([v, h], dim = -1)

# class GrowthFlowNetTrainBase(FlowNetTrainBase):
#     def __init__(
#         self,
#         growth_net,
#         *args,
#         **kwargs,
#     ):
#         super().__init__(*args, **kwargs)
#         self.growth_net = growth_net
    
#     def forward(self, x, c, t):
#         return self.flow_net(x, c, t), self.growth_net(x, c, t)

#     def _compute_loss(self, batch):

#         x0, x1, t0, t1, cond = self._prepare_batch(batch)
#         t0 = self.normalize_time(t0)
#         t1 = self.normalize_time(t1)

#         loss = 0

#         for i in range(x0.shape[0]):
        
#             t, xt, ut, gt, log_etat = self.flow_matcher.sample_location_and_conditional_flow(x0[i], x1[i], t0[i], t1[i],
#                                                                                             self.get_device())

#             c = cond[i].unsqueeze(0).repeat(x0[i].shape[0], 1)
    
#             device = self.get_device()
#             mask = (torch.rand(c.shape[0], 1) > self.config.cfg_p_u).int().to(device)
#             vt, ht = self(xt, c * mask, t)
            
#             weight = torch.exp(log_etat).unsqueeze(1)
#             loss += torch.mean((vt - ut) ** 2 * weight)
#             loss += torch.mean((ht.squeeze(1) - gt) ** 2 * weight)


#         return loss / x0.shape[0]

#     def sample_traj_and_weight(self, x, c, t0, t1, num_samples, steps=2000):

#         x /= self.sample_rescale

#         device = self.get_device()
#         t0 = self.normalize_time(t0)
#         t1 = self.normalize_time(t1)
    
#         node = NeuralODE(WrappedVectorFieldGrowth(self.flow_net, self.growth_net, 
#                                             c, self.config.cfg_w), solver="dopri5", sensitivity="adjoint")
#         indices = np.random.choice(x.shape[0], num_samples, replace=False)
#         indices = torch.from_numpy(indices).to(device)
#         x_g = np.concatenate([x[indices], np.zeros((num_samples, 1))], axis = 1) #intialize log weights at zero
                             
#         with torch.no_grad():
#             traj = node.trajectory(
#                 torch.from_numpy(x_g).float().to(device),
#                 t_span=torch.linspace(t0, t1, steps),
#             ).cpu()

#         traj[-1,:,:-1] *= self.sample_rescale
#         return traj 

#     def sample_and_weight(self, x, c, t0, t1, num_samples, steps=2000):
#         traj = self.sample_traj_and_weight(x, c, t0, t1, num_samples, steps)
#         xs = traj[-1,:,:-1]
#         ws = traj[-1,:,-1]
#         ws = F.softmax(ws)
#         return xs, ws
    
#     def sample_traj(self, x, c, t0, t1, num_samples, steps=2000):

#         traj = self.sample_traj_and_weight(x, c, t0, t1, num_samples, steps)
#         return traj[:,:,:-1]

#     def sample(self, x, c, t0, t1, num_samples, steps=2000):
#         traj = self.sample_traj(x, c, t0, t1, num_samples, steps)
#         return traj[-1]

############################################################################################################################
#METRIC?
############################################################################################################################

class MetricFlowNetTrainBase(FlowNetTrainBase):
    def __init__(
        self,
        geo_net,
        embed_net,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.geo_net = geo_net
        self.embed_net = embed_net
        ### Only have these parameters to load them onto device

    def _compute_loss(self, batch):

        x0, x1, t0, t1 = self._prepare_batch(batch)
        t0 = self.normalize_time(t0)
        t1 = self.normalize_time(t1)
        loss = 0

        for i in range(x0.shape[0]):
            
            t, xt, dxt, _, _, _ = self.flow_matcher.sample_location_and_conditional_flow(x0[i], x1[i], t0[i], t1[i])
            
            vt = self(xt.detach(), t) #TODO: the main inefficiency, we should only do forward once outside the loop
    
            loss += torch.mean((vt - dxt.detach()) ** 2)

        return loss / x0.shape[0]