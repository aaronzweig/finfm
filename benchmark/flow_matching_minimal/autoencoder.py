import numpy as np
import torch
from torch import nn
from torch.nn.utils import spectral_norm
import pytorch_lightning as pl

# EXTREME LAZY FIX FOR PATH MANAGEMENT
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from models.modules import SimpleDenseNet

class PointCloudDataset(torch.utils.data.Dataset):
    """Simple dataset that keeps the full distance matrix for batching."""

    def __init__(self, pointcloud, distances):
        self.pointcloud = torch.tensor(pointcloud, dtype=torch.float32)
        self.distances = torch.tensor(distances, dtype=torch.float32)

    def __len__(self):
        return len(self.pointcloud)

    def __getitem__(self, idx):
        return self.pointcloud[idx], idx


def make_custom_collate_fn(dataset):
    """Collate function that also extracts the upper triangular distances for the batch."""

    def _collate(batch):
        points, idxs = zip(*batch)
        points = torch.stack(points)
        idx_tensor = torch.tensor(idxs)
        dist_mat = dataset.distances[idx_tensor][:, idx_tensor]
        dist_upper = dist_mat[np.triu_indices(dist_mat.size(0), k=1)]
        return {"x": points, "d": dist_upper}

    return _collate

class Autoencoder(pl.LightningModule):
    def __init__(
        self,
        data_dim,
        latent_dim,
        encoder_layer_width=None,
        dropout=0.0,
        use_spectral_norm=False,
        mean=0,
        std=1,
        dist_std=1.0,
        dist_mse_decay=0.0,
        weights_dist=1.0,
        weights_reconstr=1.0,
        weights_cycle=0.0,
        weights_cycle_dist=0.0,
        lr=1e-3,
        weight_decay=1e-5,
    ):
        super().__init__()

        super().__init__()
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.as_tensor(std, dtype=torch.float32))
        self.register_buffer("dist_std", torch.as_tensor(dist_std, dtype=torch.float32))
        self.dist_mse_decay = dist_mse_decay
        self.mlp = SimpleDenseNet(
            input_dim=data_dim,
            output_dim=latent_dim,
            hidden_dims=encoder_layer_width or [256, 128, 64],
            dropout=dropout,
            layer_norm=True,
            use_spectral_norm=use_spectral_norm
        )

        self.weights_dist = weights_dist
        self.weights_reconstr = weights_reconstr
        self.weights_cycle = weights_cycle
        self.weights_cycle_dist = weights_cycle_dist
        self.lr = lr
        self.weight_decay = weight_decay
        self.save_hyperparameters()

    def _normalize(self, x):
        return (x - self.mean) / self.std

    def _normalize_dist(self, d):
        return d / self.dist_std

    def forward(self, x, normalize=True):
        if normalize:
            x = self._normalize(x)
        return self.mlp(x)

    def distance_loss(self, dist_gt_normalized, z):
        if torch.backends.mps.is_available():
            dist_emb = torch.cdist(z, z, p=2)
            dist_emb = dist_emb[np.triu_indices(dist_emb.size(0), k=1)].flatten()
        else:
            dist_emb = torch.nn.functional.pdist(z)

        if self.dist_mse_decay > 0.0:
            return (
                torch.square(dist_emb - dist_gt_normalized)
                * torch.exp(-self.dist_mse_decay * dist_gt_normalized)
            ).mean()
        return torch.nn.functional.mse_loss(dist_emb, dist_gt_normalized)

    def _loss(self, batch, stage):
        x = batch["x"]
        d = batch["d"]
        x_norm = self._normalize(x)
        d_norm = self._normalize_dist(d)
        z = self(x_norm)

        loss = self.weights_dist * self.distance_loss(d_norm, z)
        self.log(f"{stage}/loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._loss(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._loss(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._loss(batch, "test")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
