import torch
import numpy as np

from .base_model import *
from utils.frozen import *
import torch.nn.functional as F
from torch.func import jvp, vjp
from torch.distributions import Categorical


class MetricNetTrainBase(ModelBase):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
    
    def _prepare_batch(self, batch):

        device = self.get_device()

        x, y = batch
        x = x.to(device)
        y = y.to(device)
        x = self.normalize(x)

        return x, y
    
    def training_step(self, batch, batch_idx):
        loss = self._compute_loss(batch)

        self.log("train_loss_metric", loss)
        return loss

class MetricNetCFM(MetricNetTrainBase):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

    def forward(self, x, v):
        return torch.norm(v, dim = -1)

    def _compute_loss(self, batch):
        return torch.tensor(0.0, device=self.device, requires_grad=True)
    
    def configure_optimizers(self):
        return None

from sklearn.cluster import KMeans
class MetricNetMFM(MetricNetTrainBase):
    def __init__(
        self,
        K = 20,
        kappa = 1.0,
        alpha = 1.0,
        epsilon = 1e-2,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.K = K
        self.clustering_model = KMeans(n_clusters=self.K)
        self.kappa = kappa
        self.W = torch.nn.Parameter(torch.rand(self.K, 1))
        self.train_dataloader = None  # Will be set externally before training

        self.alpha = alpha
        self.epsilon = epsilon

    def M(self, x):
        dist2 = torch.cdist(x, self.C) ** 2
        self.phi_x = torch.exp(-0.5 * self.lamda[None, :, :] * dist2[:, :, None])
        h_x = (self.W.to(x.device) * self.phi_x).sum(dim=1)
        
        M_x = 1 / (h_x + self.epsilon) ** self.alpha

        return M_x.squeeze(-1)
    
    def forward(self, x, v):
        return torch.norm(v, dim = -1) * torch.sqrt(self.M(x))

    def on_before_zero_grad(self, *args, **kwargs):
        self.W.data = torch.clamp(self.W.data, min=0.0001)

    def on_train_start(self):
        with torch.no_grad():

            data_to_fit = []
            for batch in self.train_dataloader:
                x, y = self._prepare_batch(batch)
                data_to_fit.append(x.detach().cpu())
            data_to_fit = torch.cat(data_to_fit)

            print("Fitting Clustering model...")
            self.clustering_model.fit(data_to_fit)

            clusters = self.clustering_model.cluster_centers_

            C = torch.tensor(clusters, dtype=torch.float32).to(self.device)
            self.register_buffer("C", C)
            labels = self.clustering_model.labels_
            sigmas = np.zeros((self.K, 1))

            for k in range(self.K):
                points = data_to_fit[labels == k, :]
                variance = ((points - clusters[k]) ** 2).mean(axis=0)
                sigmas[k, :] = np.sqrt(variance.mean())

            lamda = torch.tensor(
                0.5 / (self.kappa * sigmas + 1e-8) ** 2, dtype=torch.float32
            ).to(self.device)
            self.register_buffer("lamda", lamda)

    def _compute_loss(self, batch):
        x, y = self._prepare_batch(batch)
        loss = ((1 - self.M(x)) ** 2).mean()
        return loss

class MetricNetGAGA(MetricNetTrainBase):
    def __init__(
        self,
        encoder,
        discriminator,
        disc_factor=5.0,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.encoder = encoder
        self.discriminator = discriminator
        self.disc_factor = disc_factor

    def _fn(self, x):
        # x is already ModelBase-normalized (done by embed model's _prepare_batch).
        # Metric pullback of [f(x), β·s(x)] where f=encoder, s=off-manifold probability.
        # Both f and s operate on the same ambient (normalized) x.
        z = self.encoder(x, normalize=False)
        s = 1 - self.discriminator.positive_prob(x, normalize=False)  # ≈0 on-manifold, ≈1 off-manifold
        return torch.cat([z, self.disc_factor * s.unsqueeze(1)], dim=1)

    def forward(self, x, v):
        f_x, Jf_x_v = jvp(self._fn, (x,), (v,))
        return torch.norm(Jf_x_v, dim = -1)

    def _compute_loss(self, batch):
        return torch.tensor(0.0, device=self.device, requires_grad=True)

    def configure_optimizers(self):
        return None


class FinslerMixin:
    def __init__(
        self,
        tree,
        temp,
        classifier_model,
        lamb = 1.0,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.eps = 1e-8
        self.temp = temp
        self.classifier_model = classifier_model
        self.lamb = lamb
        self.tree = self._pad_tree(tree)
    
    def classifier_fn(self, x):
        return torch.softmax(self.classifier_model(x) / self.temp, dim=-1)
    
    def logit_fn(self, x):
        return self.classifier_model(x)

    def _pad_tree(self, tree):
        target = tree.shape[-1] + 1
        padded = torch.zeros((target, target))
        rows = min(target, tree.shape[0])
        cols = min(target, tree.shape[1])
        padded[:rows, :cols] = tree[:rows, :cols]
        return padded

    def fisher_rao(self, x):
        p = self.classifier_fn(x)
        diag = 1 / (p + self.eps)
        return diag
    
    def riemann_norm(self, x, v):
        return super().forward(x, v)
    
    # # Forward mode basic
    # def forward(self, x, v):
    #     riemann_term = super().forward(x, v)
    #     f_x, Jf_x_v = jvp(self.classifier_fn, (x,), (v,))
        
    #     u = 1 - f_x @ (self.tree.to(self.get_device()) + torch.eye(f_x.shape[-1], device=x.device))

    #     # D = self.fisher_rao(x)
    #     D = torch.ones_like(f_x)

    #     finsler_term = torch.sum(Jf_x_v * D * u, dim=-1)
    #     finsler_term *= riemann_term / torch.norm(v, dim=-1)
    #     return riemann_term + self.lamb * F.relu(finsler_term)

    # # Backward mode basic
    # def forward(self, x, v):
    #     riemann_term = self.riemann_norm(x, v)
    #     f_x = self.classifier_fn(x)

    #     u = 1 - f_x @ (self.tree.to(self.get_device()) + torch.eye(f_x.shape[-1], device=x.device))

    #     _, vjp_fn = vjp(self.classifier_fn, x) 
    #     h = vjp_fn(u)[0]

    #     #Crummy approximation to the dual norm
    #     # h_scale = self.riemann_norm(x, h) / torch.norm(h, dim=-1) ** 2
    #     # h = h * h_scale.unsqueeze(-1)

    #     # h_scale = self.riemann_norm(x, h) / torch.norm(h, dim=-1)
    #     # ent = torch.sum(-f_x * torch.log(f_x), dim=-1)
        
    #     # D = self.fisher_rao(x)

    #     finsler_term = torch.sum(v * h, dim=-1)
    #     return riemann_term + self.lamb * F.relu(finsler_term)

    # Classwise
    def forward(self, x, v):
        riemann_term = super().forward(x, v)
        f_x, Jf_x_v = jvp(self.classifier_fn, (x,), (v,))
        
        u = 1 - (self.tree.to(self.get_device()) + torch.eye(f_x.shape[-1], device=x.device))

        # D = 1 / (f_x + self.eps)
        D = 1

        finsler_term = torch.sum(f_x * F.relu(D * Jf_x_v @ u.T), dim=-1)

        scale = riemann_term / torch.norm(v, dim = -1)

        return riemann_term + self.lamb * scale * F.relu(finsler_term)

    # Classwise logits
    # def forward(self, x, v):
    #     riemann_term = super().forward(x, v)
    #     _, Jl_x_v = jvp(self.logit_fn, (x,), (v,))
    #     f_x = self.classifier_fn(x)
        
    #     u = 1 - (self.tree.to(self.get_device()) + torch.eye(f_x.shape[-1], device=x.device))

    #     finsler_term = torch.sum(f_x * F.relu(Jl_x_v @ u.T), dim=-1)

    #     return riemann_term + self.lamb * F.relu(finsler_term)

    # Logits
    # def forward(self, x, v):
    #     riemann_term = self.riemann_norm(x, v)
    #     _, Jl_x_v = jvp(self.logit_fn, (x,), (v,))
    #     f_x = self.classifier_fn(x)

    #     u = 1 - f_x @ (self.tree.to(self.get_device()) + torch.eye(f_x.shape[-1], device=x.device))

    #     D = f_x
        
    #     finsler_term = torch.sum(Jl_x_v * D * u, dim=-1)
    #     return riemann_term + self.lamb * F.relu(finsler_term)

    # Riemann Conformal
    # def forward(self, x, v):
    #     riemann_term = self.riemann_norm(x, v)
    #     f_x = self.classifier_fn(x)

    #     A = self.tree.to(self.get_device())
    #     M = A + A.T + torch.eye(f_x.shape[-1], device=x.device)

    #     conformal = torch.sum(f_x * (f_x @ (1 - M)), dim=-1)
        
    #     return riemann_term + self.lamb * torch.norm(v, dim=-1) * torch.sqrt(conformal)


class FinslerCFM(FinslerMixin, MetricNetCFM):
    pass

class FinslerMFM(FinslerMixin, MetricNetMFM):
    pass

class FinslerGAGA(FinslerMixin, MetricNetGAGA):
    pass


class MetricNetSBCFM(MetricNetCFM):
    pass


class FinslerSBCFM(FinslerMixin, MetricNetSBCFM):
    pass
