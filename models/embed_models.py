import torch
import torch.nn as nn

import numpy as np

from .base_model import *
from utils.frozen import *
from models.cfm import *
import torch.nn.functional as F
from torch.func import jvp


class EmbedNetTrainBase(ModelBase):
    def __init__(
        self,
        embed_net,
        metric_model,
        geo_net,
        t_global_min,
        t_global_max,
        sample_rescale,
        *args,
        **kwargs,
    ):
        
        super().__init__(*args, **kwargs)
        self.embed_net = embed_net
        self.geo_net = geo_net
        self.metric_model = metric_model
        self.t_global_min = t_global_min
        self.t_global_max = t_global_max

        self.sample_rescale = sample_rescale.float()

        self.flow_matcher = MetricFlowMatcher(sigma=self.config.sigma,
                                              geo_fn = self.geo_fn,
                                              cost_matrix_fn = self.cost_matrix_fn,
                                              no_ot = self.config.fast_ot,
                                              )

    def embed_fn(self, x):
            return self.embed_net(x)
    
    def geo_fn(self, x0, x1, t):
        return self.geo_net(torch.cat([x0, x1], dim=-1), t)
    
    def cost_matrix_fn(self, x0, x1):
            return torch.cdist(self.embed_fn(x0), self.embed_fn(x1)) ** 2

    def F(self, x, v):
        x = x * self.sample_rescale.to(self.get_device())
        return self.metric_model(x, v)

    def normalize_time(self, t):
        return (t - self.t_global_min) / (self.t_global_max - self.t_global_min)

    def _prepare_batch(self, batch):

        device = self.get_device()

        x0, x1, t0, t1 = batch
        x0.to(device)
        x1.to(device)
        t0.to(device)
        t1.to(device)
        
        return x0, x1, t0, t1

    def embed_loss(self, x, v):
        df_x_v = jvp(self.embed_fn, (x,), (v,))[1]
        norm_diff = torch.abs(torch.norm(df_x_v, dim=-1) - self.F(x, v))
        loss = torch.mean(norm_diff)
        return loss
    
    def geo_loss(self, x, v):
        return 0.5 * torch.mean(self.F(x, v) ** 2)

    def _compute_loss(self, batch):
        x0, x1, t0, t1 = self._prepare_batch(batch)
        t0 = self.normalize_time(t0)
        t1 = self.normalize_time(t1)
        loss_embed, loss_geo = 0, 0

        for i in range(x0.shape[0]):
            t, xt, dxt = self.flow_matcher.sample_location_and_conditional_flow(x0[i], x1[i], t0[i], t1[i],
                                                                                ot_sample=self.config.ot_in_embed)
            xt_free, dxt_free = xt.detach(), dxt.detach()
            v = torch.randn_like(dxt_free)
            v /= torch.norm(v, dim=-1, keepdim=True)

            #TODO: should we normalize dxt_free too?  Morally yes because we're comparing two homogeneous norms
            loss_embed += self.embed_loss(xt_free, dxt_free)
            loss_embed += self.embed_loss(xt_free, v)
            loss_geo += self.geo_loss(xt, dxt)

        loss_embed /= torch.max(self.sample_rescale) ** 2
        loss_geo /= torch.max(self.sample_rescale) ** 2

        return loss_embed / x0.shape[0], loss_geo / x0.shape[0]

    def training_step(self, batch, batch_idx):
        loss_embed, loss_geo = self._compute_loss(batch)

        self.log("train_loss_embed", loss_embed)
        self.log("train_loss_geo", loss_geo)
        return loss_embed + loss_geo

    def sample_geodesic(self, batch, points = 50, ot_sample=True):

        old_sigma = self.flow_matcher.sigma
        self.flow_matcher.sigma = 0

        x0, x1, t0, t1 = self._prepare_batch(batch)
        t0 = self.normalize_time(t0)
        t1 = self.normalize_time(t1)

        paths = []
        
        i = 0
        x0_, x1_ = x0[i], x1[i]

        x0_ /= self.sample_rescale
        x1_ /= self.sample_rescale
        
        if ot_sample:
            x0_, x1_ = self.flow_matcher.ot_sample(x0_, x1_) #Freeze a sample of points from coupling

        for j in np.linspace(0, 1, points):
            t = torch.tensor(j)
            t = t.unsqueeze(0).repeat(x0[i].shape[0])
            t.requires_grad_(True)
            _, xt, _ = self.flow_matcher.sample_location_and_conditional_flow(x0_, x1_, t0[i], t1[i], 
                                                                                    t=t, ot_sample=False)
            paths.append(xt.detach() * self.sample_rescale)

        self.flow_matcher.sigma = old_sigma
        return torch.stack(paths, dim = 0)
    

class FinslerEmbedNetTrainBase(EmbedNetTrainBase):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        
        super().__init__(*args, **kwargs)
        self.beta = nn.Parameter(torch.randn(self.config.latent_dim//2))


    #TODO: this really only makes sense if skip = False
    def embed_fn(self, x):
        return self.embed_net(x)
    
    def embed_loss(self, x, v):
        df_x_v = jvp(self.embed_fn, (x,), (v,))[1]
        dphi, dpsi = df_x_v[:, self.beta.shape[0]:], df_x_v[:, :self.beta.shape[0]] @ -self.beta
        norm_diff = torch.abs(torch.norm(dphi, dim=-1) + dpsi - self.F(x, v))
        loss = torch.mean(norm_diff)
        return loss 

    def cost_matrix_fn(self, x0, x1):
        z0 = self.embed_fn(x0)
        z1 = self.embed_fn(x1)
        phi0, psi0 = z0[:, self.beta.shape[0]:], z0[:, :self.beta.shape[0]] @ self.beta
        phi1, psi1 = z1[:, self.beta.shape[0]:], z1[:, :self.beta.shape[0]] @ self.beta
        M = torch.cdist(phi0, phi1)
        M += F.relu(psi0.unsqueeze(1) - psi1.unsqueeze(0))
        return M ** 2