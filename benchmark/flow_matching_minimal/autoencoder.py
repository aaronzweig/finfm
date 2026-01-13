import numpy as np
import torch
from torch import nn
from torch.nn.utils import spectral_norm
import pytorch_lightning as pl


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


def split_train_val_test(x, test_size=0.1, val_size=0.1):
    perm = np.random.permutation(len(x))
    test_n = int(len(x) * test_size)
    val_n = int(len(x) * val_size)
    test_idx = perm[:test_n]
    val_idx = perm[test_n : test_n + val_n]
    train_idx = perm[test_n + val_n :]
    return train_idx, val_idx, test_idx


class MLP(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        layer_widths=None,
        activation="relu",
        batch_norm=False,
        dropout=0.0,
        use_spectral_norm=False,
    ):
        super().__init__()
        layer_widths = layer_widths or [256, 128, 64]
        act_map = {
            "relu": nn.ReLU(),
            "leaky_relu": nn.LeakyReLU(),
            "tanh": nn.Tanh(),
        }
        if activation not in act_map:
            raise ValueError(f"Unknown activation {activation}")

        layers = []
        for i, width in enumerate(layer_widths):
            in_features = in_dim if i == 0 else layer_widths[i - 1]
            linear = nn.Linear(in_features, width)
            if use_spectral_norm:
                linear = spectral_norm(linear)
            layers.append(linear)
            if batch_norm:
                layers.append(nn.BatchNorm1d(width))
            layers.append(act_map[activation])
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

        final = nn.Linear(layer_widths[-1], out_dim)
        if use_spectral_norm:
            final = spectral_norm(final)
        layers.append(final)
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Encoder(nn.Module):
    def __init__(
        self,
        data_dim,
        latent_dim,
        layer_widths=None,
        activation="relu",
        batch_norm=False,
        dropout=0.0,
        use_spectral_norm=False,
        mean=0,
        std=1,
        dist_std=1.0,
        dist_mse_decay=0.0,
    ):
        super().__init__()
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.as_tensor(std, dtype=torch.float32))
        self.register_buffer("dist_std", torch.as_tensor(dist_std, dtype=torch.float32))
        self.dist_mse_decay = dist_mse_decay
        self.mlp = MLP(
            data_dim,
            latent_dim,
            layer_widths=layer_widths or [256, 128, 64],
            activation=activation,
            batch_norm=batch_norm,
            dropout=dropout,
            use_spectral_norm=use_spectral_norm,
        )

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


class Decoder(nn.Module):
    def __init__(
        self,
        latent_dim,
        data_dim,
        layer_widths=None,
        activation="relu",
        batch_norm=False,
        dropout=0.0,
        use_spectral_norm=False,
        mean=0,
        std=1,
    ):
        super().__init__()
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.as_tensor(std, dtype=torch.float32))
        self.mlp = MLP(
            latent_dim,
            data_dim,
            layer_widths=layer_widths or [64, 128, 256],
            activation=activation,
            batch_norm=batch_norm,
            dropout=dropout,
            use_spectral_norm=use_spectral_norm,
        )

    def _unnormalize(self, x):
        return x * self.std + self.mean

    def forward(self, z, unnormalize=False):
        x = self.mlp(z)
        return self._unnormalize(x) if unnormalize else x

    @staticmethod
    def reconstruction_loss(x_norm, xhat_norm):
        return torch.nn.functional.mse_loss(xhat_norm, x_norm)


class Autoencoder(pl.LightningModule):
    def __init__(
        self,
        data_dim,
        latent_dim,
        encoder_layer_width=None,
        decoder_layer_width=None,
        activation="relu",
        batch_norm=False,
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
        self.encoder = Encoder(
            data_dim,
            latent_dim,
            layer_widths=encoder_layer_width or [256, 128, 64],
            activation=activation,
            batch_norm=batch_norm,
            dropout=dropout,
            use_spectral_norm=use_spectral_norm,
            mean=mean,
            std=std,
            dist_std=dist_std,
            dist_mse_decay=dist_mse_decay,
        )
        self.decoder = Decoder(
            latent_dim,
            data_dim,
            layer_widths=decoder_layer_width or [64, 128, 256],
            activation=activation,
            batch_norm=batch_norm,
            dropout=dropout,
            use_spectral_norm=use_spectral_norm,
            mean=mean,
            std=std,
        )

        self.weights_dist = weights_dist
        self.weights_reconstr = weights_reconstr
        self.weights_cycle = weights_cycle
        self.weights_cycle_dist = weights_cycle_dist
        self.lr = lr
        self.weight_decay = weight_decay
        self.save_hyperparameters()

    def forward(self, x):
        return self.decoder(self.encoder(x, normalize=True), unnormalize=False)

    def _loss(self, batch, stage):
        x = batch["x"]
        d = batch["d"]
        x_norm = self.encoder._normalize(x)
        d_norm = self.encoder._normalize_dist(d)
        z = self.encoder(x_norm)
        xhat_norm = self.decoder(z, unnormalize=False)

        loss = 0.0
        if self.weights_dist > 0:
            dist_loss = self.encoder.distance_loss(d_norm, z)
            loss = loss + self.weights_dist * dist_loss
            self.log(f"{stage}/dist_loss", dist_loss, prog_bar=True, on_epoch=True)

        if self.weights_reconstr > 0:
            recon_loss = self.decoder.reconstruction_loss(x_norm, xhat_norm)
            loss = loss + self.weights_reconstr * recon_loss
            self.log(f"{stage}/reconstr_loss", recon_loss, prog_bar=True, on_epoch=True)

        if self.weights_cycle + self.weights_cycle_dist > 0:
            z_cycle = self.encoder(xhat_norm, normalize=False)
            if self.weights_cycle > 0:
                cycle_loss = torch.nn.functional.mse_loss(z, z_cycle)
                loss = loss + self.weights_cycle * cycle_loss
                self.log(f"{stage}/cycle_loss", cycle_loss, prog_bar=True, on_epoch=True)
            if self.weights_cycle_dist > 0:
                cycle_dist_loss = self.encoder.distance_loss(d_norm, z_cycle)
                loss = loss + self.weights_cycle_dist * cycle_dist_loss
                self.log(
                    f"{stage}/cycle_dist_loss",
                    cycle_dist_loss,
                    prog_bar=True,
                    on_epoch=True,
                )

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
