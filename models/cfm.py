import numpy as np
import torch
from torchcfm.optimal_transport import OTPlanSampler
from torchcfm.conditional_flow_matching import pad_t_like_x, ConditionalFlowMatcher
from torch.func import jvp
from utils.frozen import *
from models.ot_sampler import GeneralOTPlanSampler

############################################################################################################################

class OTFlowMatcher(ConditionalFlowMatcher):
    def __init__(
        self, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.ot_sampler = OTPlanSampler(method="exact")

    def compute_mu_t(self, x0, x1, t, t_min, t_max):

        t = pad_t_like_x(t, x0)
        return (t_max - t) / (t_max - t_min) * x0 + (t - t_min) / (
            t_max - t_min
        ) * x1

    def sample_xt(self, x0, x1, t, epsilon, t_min, t_max):
        mu_t = self.compute_mu_t(x0, x1, t, t_min, t_max)
        sigma_t = self.compute_sigma_t(t)
        sigma_t = pad_t_like_x(sigma_t, x0)
        return mu_t + sigma_t * epsilon

    def sample_location_and_conditional_flow(
        self,
        x0,
        x1,
        t_min,
        t_max,
        t=None,
    ):

        if t is None:
            t = torch.rand(x0.shape[0])
        t = t.type_as(x0)
        t = t * (t_max - t_min) + t_min

        x0, x1 = self.ot_sampler.sample_plan(x0, x1)

        eps = self.sample_noise_like(x0)
        xt = self.sample_xt(x0, x1, t, eps, t_min, t_max)
        ut = self.compute_conditional_flow(x0, x1, t, xt, t_min, t_max)

        return t, xt, ut

    def compute_conditional_flow(self, x0, x1, t, xt, t_min, t_max):
        del xt
        t = pad_t_like_x(t, x0)
        return (x1 - x0) / (t_max - t_min)


############################################################################################################################


class UOTPlanSampler(OTPlanSampler):

    #TODO: sanity check this?  The theory uses pi directly, but I think rescaling shouldn't change much?
    def sample_plan(self, x0, x1, replace=True):
        pi = self.get_map(x0, x1)
        i, j = self.sample_map(pi, x0.shape[0], replace=replace)
        pi_tilde = pi / np.sum(pi)
        r0 = -np.log(np.sum(pi_tilde, axis = 1)) - np.log(x0.shape[0])
        r1 = -np.log(np.sum(pi_tilde, axis = 0)) - np.log(x0.shape[0])
        
        return x0[i], x1[j], r0[i], r1[j]

class UOTFlowMatcher(OTFlowMatcher):
    def __init__(
        self, reg, reg_m, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.ot_sampler = UOTPlanSampler(method="unbalanced", reg=reg, reg_m=reg_m)
        # self.ot_sampler = UOTPlanSampler(method="exact", reg=reg, reg_m=reg_m) #DEBUG


    def sample_location_and_conditional_flow(
        self,
        x0,
        x1,
        t_min,
        t_max,
        device,
        t=None,
    ):

        if t is None:
            t = torch.rand(x0.shape[0])
        t = t.type_as(x0)
        t = t * (t_max - t_min) + t_min

        x0, x1, r0, r1 = self.ot_sampler.sample_plan(x0, x1)
        r0 = torch.from_numpy(r0).to(device)
        r1 = torch.from_numpy(r1).to(device)

        eps = self.sample_noise_like(x0)
        xt = self.sample_xt(x0, x1, t, eps, t_min, t_max)
        ut = self.compute_conditional_flow(x0, x1, t, xt, t_min, t_max)
        gt = self.compute_conditional_growth(r0, r1, t_min, t_max)
        log_etat = self.compute_log_eta(r0, r1, t, t_min, t_max)

        

        return t, xt, ut, gt, log_etat


    def compute_conditional_growth(self, r0, r1, t_min, t_max):
        return (r1 - r0) / (t_max - t_min)

    def compute_log_eta(self, r0, r1, t, t_min, t_max):
        return (t_max - t) / (t_max - t_min) * r0 + (t - t_min) / (
            t_max - t_min
        ) * r1

############################################################################################################################

class MetricFlowMatcher(OTFlowMatcher):
    def __init__(
        self, 
        geo_fn, 
        cost_matrix_fn,
        no_ot = False, 
        *args, 
        **kwargs):
        super().__init__(*args, **kwargs)
        self.ot_sampler = GeneralOTPlanSampler(method="exact", cost_matrix_fn=cost_matrix_fn)
        self.geo_fn = geo_fn
        self.no_ot = no_ot

    def gamma(self, t, t_min, t_max):
        return (
            1.0
            - ((t - t_min) / (t_max - t_min)) ** 2
            - ((t_max - t) / (t_max - t_min)) ** 2
        )

    def compute_mu_t(self, x0, x1, t, t_min, t_max):

        t = pad_t_like_x(t, x0)

        return (
            (t_max - t) / (t_max - t_min) * x0
            + (t - t_min) / (t_max - t_min) * x1
            + self.gamma(t, t_min, t_max) * self.geo_fn(x0, x1, t)
        )  

    def ot_sample(self, x0, x1):
        if self.no_ot:
            return x0, x1
        
        pi = self.ot_sampler.get_map(x0, x1)
        i, j = self.ot_sampler.sample_map(pi, x0.shape[0], replace=True)
        x0, x1 = x0[i], x1[j]
        
        return x0, x1
    
    def sample_location_and_conditional_flow(
        self,
        x0,
        x1,
        t_min,
        t_max,
        t=None,
        ot_sample=True,
    ):
        if t is None:
            t = torch.rand(x0.shape[0], requires_grad=True)

        t = t.type_as(x0)
        t = t * (t_max - t_min) + t_min

        if ot_sample:
            x0, x1 = self.ot_sample(x0, x1)

        eps = self.sample_noise_like(x0)
        xt, ut = self.compute_xt_and_conditional_flow(x0, x1, t, eps, t_min, t_max)
        return t, xt, ut

    def compute_xt_and_conditional_flow(self, x0, x1, t, eps, t_min, t_max):
        t = pad_t_like_x(t, x0)

        def f(tt):
            t_padded = pad_t_like_x(tt, x0)
            return self.sample_xt(x0, x1, t_padded, eps, t_min, t_max)

        xt, ut = jvp(f, (t,), (torch.ones_like(t),))
        return xt, ut.squeeze(-1)