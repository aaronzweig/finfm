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
        *args,
        **kwargs,
    ):
        
        super().__init__(*args, **kwargs)
        self.embed_net = embed_net
        self.geo_net = geo_net
        self.metric_model = metric_model
        self.t_global_min = t_global_min
        self.t_global_max = t_global_max

        self.flow_matcher = MetricFlowMatcher(sigma=self.config.sigma,
                                              method=self.config.method,
                                              reg=self.config.reg,
                                              reg_m=self.config.reg_m,
                                              geo_fn = self.geo_fn,
                                              cost_matrix_fn = self.cost_matrix_fn)
        
        print("DEBUG: trying dumb first loss as scaling for embed")
        self.embed_loss_scalar = torch.tensor(-1)
        self.geo_loss_scalar = torch.tensor(-1)
        
    def embed_fn(self, x):
            if self.config.no_learning or self.config.mfm.use_euclidean_ot:
                return x
            return self.embed_net(x)
    
    def geo_fn(self, x0, x1, t):
        if self.config.no_learning:
            return 0
        return self.geo_net(torch.cat([x0, x1], dim=-1), t)
    
    def cost_matrix_fn(self, x0, x1):
            return torch.cdist(self.embed_fn(x0), self.embed_fn(x1)) ** 2

    def F(self, x, v):
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
        x0 = self.normalize(x0)
        x1 = self.normalize(x1)
        t0 = self.normalize_time(t0)
        t1 = self.normalize_time(t1)
        
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
        loss_embed, loss_geo = 0, 0

        for i in range(x0.shape[0]):
            t, xt, dxt, log_etat = self.flow_matcher.sample_location_and_conditional_flow(x0[i], x1[i], t0[i], t1[i],
                                                                                ot_sample=self.config.ot_in_embed,
                                                                                time_per_batch=self.config.time_per_batch)
            xt_free, dxt_free = xt.detach(), dxt.detach()
            # v = torch.randn_like(dxt_free)
            # v /= torch.norm(v, dim=-1, keepdim=True)

            #TODO: should we normalize dxt_free too?  Morally yes because we're comparing two homogeneous norms
            loss_embed += self.embed_loss(xt_free, dxt_free)
            # loss_embed += self.embed_loss(xt_free, v)
            loss_geo += self.geo_loss(xt, dxt)

        # if self.embed_loss_scalar == -1:
        #     self.embed_loss_scalar = (loss_embed / x0.shape[0]).detach()        
        # if self.geo_loss_scalar == -1:
        #     self.geo_loss_scalar = (loss_geo / x0.shape[0]).detach()

        # return loss_embed / x0.shape[0] / self.embed_loss_scalar, loss_geo / x0.shape[0] / self.geo_loss_scalar

        return loss_embed / x0.shape[0], loss_geo / x0.shape[0]

    def training_step(self, batch, batch_idx):
        loss_embed, loss_geo = self._compute_loss(batch)

        self.log("train_loss_embed", loss_embed)
        self.log("train_loss_geo", loss_geo)
        return loss_embed + loss_geo

    def _sample_geodesic(self, batch, timepoints, ot_sample=True, weighted=False):

        old_sigma = self.flow_matcher.sigma
        self.flow_matcher.sigma = 0

        x0, x1, t0, t1 = self._prepare_batch(batch)
        assert x0.shape[0] == 1

        paths = []
        weights = []
        
        x0_, x1_ = x0[0], x1[0]
        t0_, t1_ = t0[0], t1[0]
        r0_, r1_ = None, None
        if ot_sample:
            x0_, x1_, r0_, r1_ = self.flow_matcher.ot_sampler.sample_plan(x0_, x1_) #Freeze a sample of points from coupling

        for j in timepoints:
            t = torch.tensor(j)
            t = t.unsqueeze(0).repeat(x0_.shape[0])
            t.requires_grad_(True)
            _, xt, _, log_etat = self.flow_matcher.sample_location_and_conditional_flow(x0_, x1_, t0_, t1_, 
                                                                                    t=t, 
                                                                                    r0=r0_,
                                                                                    r1=r1_,
                                                                                    ot_sample=False)
            xt = self.unnormalize(xt)
            paths.append(xt.detach())
            weights.append(log_etat.detach())

        self.flow_matcher.sigma = old_sigma
        if weighted:
            return torch.stack(paths, dim = 0), torch.stack(weights, dim = 0)
        return torch.stack(paths, dim = 0)
    
    def sample_geodesic_path(self, batch, num_points, ot_sample=True, weighted=False):
        timepoints = np.linspace(0, 1, num_points)
        return self._sample_geodesic(batch, timepoints, ot_sample, weighted)

    def sample_geodesic_time(self, batch, t, ot_sample=True, weighted=False):
        _, _, t0, t1 = self._prepare_batch(batch)
        t0 = t0[0]
        t1 = t1[0]
        timepoints = torch.Tensor([t])
        timepoints = self.normalize_time(timepoints)
        timepoints = (timepoints - t0) / (t1 - t0)

        if weighted:
            paths, weights = self._sample_geodesic(batch, timepoints, ot_sample, weighted)
            return paths[0], weights[0]
        paths = self._sample_geodesic(batch, timepoints, ot_sample, weighted)
        return paths[0]

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