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
                                              embed_fn = self.embed_fn,
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

    def _compute_loss(self, batch):
        x0, x1, t0, t1 = self._prepare_batch(batch)
        t0 = self.normalize_time(t0)
        t1 = self.normalize_time(t1)
        loss = 0

        for i in range(x0.shape[0]):
            t, xt, dxt = self.flow_matcher.sample_location_and_conditional_flow(x0[i], x1[i], t0[i], t1[i],
                                                                                ot_sample=self.config.ot_in_embed)
            xt_free, dxt_free = xt.detach(), dxt.detach()
            df_xt = jvp(self.embed_fn, (xt_free,), (dxt_free,))[1]

            #TODO: include random velocity sampling
            norm_diff = torch.abs(torch.norm(df_xt, dim=-1) - self.F(xt_free, dxt_free))
            loss += torch.mean(norm_diff)

            loss += 0.5 * torch.mean(self.F(xt, dxt) ** 2)

        loss /= torch.max(self.sample_rescale) ** 2
            
        return loss / x0.shape[0]

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

        #TODO: order wrong? need beta defined to make embed_fn and then define flow_matcher
        self.beta = nn.Parameter(torch.randn(self.embed_net.latent_dim//2))

    #TODO: this logic assumes skip = False in the EmbedNet
    def embed_fn(self, x):
        z = self.embed_net(x)
        psi, phi = z[:, :self.beta.shape[0]], z[:, self.beta.shape[0]:]
        return phi, psi @ self.beta
    
    def cost_matrix_fn(self, x0, x1):
        phi0, psi0 = self.embed_fn(x0)
        phi1, psi1 = self.embed_fn(x1)
        M = torch.cdist(phi0, phi1)
        M += F.relu(psi0.unsqueeze(1) - psi1.unsqueeze(0))
        return M ** 2