import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.nn.utils import spectral_norm
import os

# EXTREME LAZY FIX FOR PATH MANAGEMENT
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from models.modules import SimpleDenseNet

class Discriminator(pl.LightningModule):
    def __init__(
        self,
        in_dim,
        layer_widths=None,
        activation="relu",
        normalize=False,
        lr=1e-4,
        weight_decay=1e-4,
        batch_norm=True,
        dropout=0.5,
        use_spectral_norm=True,
    ):
        super().__init__()
        self.normalize = normalize
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
        if self.normalize:
            x = (x - self.mean) / self.std
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

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._step(batch, "test")

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

    def on_validation_epoch_end(self):
        self._epoch_end("val")

    def on_test_epoch_end(self):
        self._epoch_end("test")

    def positive_prob(self, x):
        return F.softmax(self(x), dim=1)[:, 1]

    def positive_score(self, x):
        return self(x)[:, 1]

    def negative_score(self, x):
        return self(x)[:, 0]