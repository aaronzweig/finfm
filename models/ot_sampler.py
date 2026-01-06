import math
import warnings
from typing import Union

import numpy as np
import torch
import os
import sys
from torchcfm.optimal_transport import OTPlanSampler
from torchcfm.conditional_flow_matching import pad_t_like_x, ConditionalFlowMatcher
from utils.frozen import *

from torchcfm.optimal_transport import OTPlanSampler
import ot as pot

class GeneralOTPlanSampler(OTPlanSampler):

    def __init__(
        self,
        cost_matrix_fn,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.cost_matrix_fn = cost_matrix_fn

    def get_map(self, x0, x1):

        a, b = pot.unif(x0.shape[0]), pot.unif(x1.shape[0])
        if x0.dim() > 2:
            x0 = x0.reshape(x0.shape[0], -1)
        if x1.dim() > 2:
            x1 = x1.reshape(x1.shape[0], -1)
        M = self.cost_matrix_fn(x0, x1)

        if self.normalize_cost:
            M = M / M.max()  # should not be normalized when using minibatches
        p = self.ot_fn(a, b, M.detach().cpu().numpy())
        if not np.all(np.isfinite(p)):
            print("ERROR: p is not finite")
            print(p)
            print("Cost mean, max", M.mean(), M.max())
            print(x0, x1)
        if np.abs(p.sum()) < 1e-8:
            if self.warn:
                warnings.warn("Numerical errors in OT plan, reverting to uniform plan.")
            p = np.ones_like(p) / p.size
        return p