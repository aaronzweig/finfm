import torch
import numpy as np

from .base_model import *
from utils.frozen import *
from torch.func import jvp, vmap, jacrev

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
        x.to(device)
        y.to(device)

        return x, y

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
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.K = K
        self.clustering_model = KMeans(n_clusters=self.K)
        self.kappa = kappa
        self.W = torch.nn.Parameter(torch.rand(self.K, 1))
        self.train_dataloader = None  # Will be set externally before training

        self.alpha = 1.0
        self.epsilon = 1e-2

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
            for batch, *_ in self.train_dataloader:
                data_to_fit.append(batch.detach().cpu())
            data_to_fit = torch.cat(data_to_fit)

            print("Fitting Clustering model...")
            self.clustering_model.fit(data_to_fit)

            clusters = self.clustering_model.cluster_centers_

            self.C = torch.tensor(clusters, dtype=torch.float32).to(self.device)
            labels = self.clustering_model.labels_
            sigmas = np.zeros((self.K, 1))

            for k in range(self.K):
                points = data_to_fit[labels == k, :]
                variance = ((points - clusters[k]) ** 2).mean(axis=0)
                sigmas[k, :] = np.sqrt(variance.mean())

            self.lamda = torch.tensor(
                0.5 / (self.kappa * sigmas + 1e-8) ** 2, dtype=torch.float32
            ).to(self.device)

    def _compute_loss(self, batch):
        x, y = self._prepare_batch(batch)
        loss = ((1 - self.M(x)) ** 2).mean()
        return loss

class FinslerMetricNet(MetricNetTrainBase):
    def __init__(
        self,
        T,
        temp,
        riemannian_metric_model,
        classifier_model,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.T = T
        self.eps = 1e-8
        self.temp = temp
        self.riemannian_metric_model = riemannian_metric_model
        self.classifier_model = classifier_model
        
    def _Jf(self, x):
        return jacrev(self.classifier_model)(x)
    
    # temp + softmax
    def M_fisher_rao(self, x):
        p = torch.softmax(self.classifier_model(x) / self.temp, dim=-1)
        diag = 1 / (p + self.eps)
        return torch.eye(p.shape[-1], device=x.device) * diag[None, :]

    def M_finsler(self, x, v):
        gx = self.riemannian_metric_model(x)
        v_gx_norm = torch.norm(v, dim=-1) * gx
        Jf_x = self._Jf(x)
        pre_simplex = (1 - (self.M_fisher_rao(x) + self.T)) @ self.classifier_model(x)
        on_simplex = Jf_x.transpose(-1, -2) @ pre_simplex[..., None]
        finsler_term = torch.inner(v, on_simplex.squeeze(-1), dim=-1)
        return v_gx_norm + torch.relu(finsler_term)
    
    def forward(self, x, v):
        return self.M_finsler(x, v)
        
        
        
