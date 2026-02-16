import numpy as np
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
import scipy.spatial as spatial

from models.base_model import ModelBase
from models.modules import SimpleDenseNet


# --- Dataset and collate for autoencoder training ---

class PointCloudDataset(Dataset):
    """Dataset that keeps the full distance matrix for batching."""

    def __init__(self, pointcloud, distances):
        self.pointcloud = torch.tensor(pointcloud, dtype=torch.float32)
        self.distances = torch.tensor(distances, dtype=torch.float32)

    def __len__(self):
        return len(self.pointcloud)

    def __getitem__(self, idx):
        return self.pointcloud[idx], idx


def make_custom_collate_fn(dataset):
    """Collate function that extracts upper triangular distances for the batch."""

    def _collate(batch):
        points, idxs = zip(*batch)
        points = torch.stack(points)
        idx_tensor = torch.tensor(idxs)
        dist_mat = dataset.distances[idx_tensor][:, idx_tensor]
        dist_upper = dist_mat[np.triu_indices(dist_mat.size(0), k=1)]
        return {"x": points, "d": dist_upper}

    return _collate


# --- Autoencoder (inherits ModelBase for shared normalization) ---

class Autoencoder(ModelBase):
    def __init__(
        self,
        config,
        latent_dim=3,
        encoder_layer_width=None,
        dropout=0.0,
        use_spectral_norm=False,
        dist_std=1.0,
        dist_mse_decay=0.0,
        weights_dist=1.0,
        ae_lr=1e-3,
        ae_weight_decay=1e-5,
    ):
        super().__init__(config=config)

        self.register_buffer("dist_std", torch.as_tensor(dist_std, dtype=torch.float32))
        self.dist_mse_decay = dist_mse_decay
        self.mlp = SimpleDenseNet(
            input_dim=config.pc_dim,
            output_dim=latent_dim,
            hidden_dims=encoder_layer_width or [256, 128, 64],
            dropout=dropout,
            layer_norm=True,
            use_spectral_norm=use_spectral_norm,
        )
        self.weights_dist = weights_dist
        self.ae_lr = ae_lr
        self.ae_weight_decay = ae_weight_decay

    def _normalize_dist(self, d):
        return d / self.dist_std

    def forward(self, x, normalize=True):
        if normalize:
            x = self.normalize(x)
        return self.mlp(x)

    def distance_loss(self, dist_gt_normalized, z):
        if torch.backends.mps.is_available():
            dist_emb = torch.cdist(z, z, p=2)
            dist_emb = dist_emb[np.triu_indices(dist_emb.size(0), k=1)].flatten()
        else:
            dist_emb = F.pdist(z)

        if self.dist_mse_decay > 0.0:
            return (
                torch.square(dist_emb - dist_gt_normalized)
                * torch.exp(-self.dist_mse_decay * dist_gt_normalized)
            ).mean()
        return F.mse_loss(dist_emb, dist_gt_normalized)

    def _loss(self, batch, stage):
        x = batch["x"]
        d = batch["d"]
        d_norm = self._normalize_dist(d)
        # forward handles normalization of x
        z = self(x)
        loss = self.weights_dist * self.distance_loss(d_norm, z)
        self.log(f"{stage}/loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._loss(batch, "train")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.ae_lr, weight_decay=self.ae_weight_decay)


# --- Discriminator (standalone pl.LightningModule, operates in latent space) ---

class Discriminator(pl.LightningModule):
    def __init__(
        self,
        in_dim,
        layer_widths=None,
        lr=1e-4,
        weight_decay=1e-4,
        dropout=0.5,
        use_spectral_norm=True,
    ):
        super().__init__()
        self.mlp = SimpleDenseNet(
            input_dim=in_dim,
            output_dim=2,
            hidden_dims=layer_widths or [256, 128, 64],
            layer_norm=True,
            dropout=dropout,
            use_spectral_norm=use_spectral_norm,
        )
        self.lr = lr
        self.weight_decay = weight_decay
        self.save_hyperparameters()
        self._cache = {"train": [], "val": [], "test": []}

    def forward(self, x):
        return self.mlp(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def _step(self, batch, stage):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self._cache[stage].append((logits.detach(), y.detach()))
        self.log(f"{stage}_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def _epoch_end(self, stage):
        if not self._cache[stage]:
            return
        logits, labels = zip(*self._cache[stage])
        logits = torch.cat(logits)
        labels = torch.cat(labels)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()
        self.log(f"{stage}_acc", acc, prog_bar=True)
        self._cache[stage].clear()

    def on_train_epoch_end(self):
        self._epoch_end("train")

    def positive_prob(self, x):
        return F.softmax(self(x), dim=1)[:, 1]

    def positive_score(self, x):
        return self(x)[:, 1]

    def negative_score(self, x):
        return self(x)[:, 0]


# --- Negative sampling utilities ---

def neg_sample_additive(x, noise_levels, seed=42):
    """Add Gaussian noise at multiple scales."""
    np.random.seed(seed)
    noisy = []
    for level in noise_levels:
        noise = np.random.randn(*x.shape)
        noisy.append(x + noise * level)
    return np.vstack(noisy)


def compute_kernel(X, Y, sigma=1.0):
    D = spatial.distance.cdist(X, Y)
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp((-D**2) / (2 * sigma**2))


def sampling_rejection(x, x_noisy, method="density", k=20, threshold=0.01):
    """
    Reject negative samples that stay too close to the data manifold.
    Returns a boolean mask (True = reject).
    """
    if method == "density":
        distances = spatial.distance.cdist(x_noisy, x)
        dist_closest = np.partition(distances, k, axis=1)[:, :k]
        dist_mean = np.mean(dist_closest, axis=1)
        return dist_mean <= threshold
    if method == "sugar":
        G_TN = compute_kernel(x, x_noisy)
        P_TN = G_TN / np.sum(G_TN, axis=0, keepdims=True)
        G_NT = G_TN.T
        P_NT = G_NT / np.sum(G_NT, axis=1, keepdims=True)
        x_noisy_bar = P_NT @ (P_TN @ x_noisy)
        change = np.linalg.norm(x_noisy - x_noisy_bar, axis=1)
        return change <= threshold
    raise ValueError(f"Invalid sampling rejection method: {method}")


# --- Encoding helper ---

def encode_data(x, encoder, device, batch_size=256):
    """Encode data through a frozen autoencoder in batches."""
    encodings = []
    encoder.eval()
    encoder.to(device)
    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            batch_slice = x[i : i + batch_size]
            if isinstance(batch_slice, torch.Tensor):
                batch = batch_slice.to(device=device, dtype=torch.float32)
            else:
                batch = torch.as_tensor(batch_slice, dtype=torch.float32, device=device)
            # normalize=True: encoder applies ModelBase normalization
            encodings.append(encoder(batch, normalize=True).cpu().numpy())
    return np.concatenate(encodings, axis=0)
