import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.nn.utils import spectral_norm
import os


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
        layers = []
        act_map = {"relu": nn.ReLU(), "leaky_relu": nn.LeakyReLU(), "tanh": nn.Tanh()}
        if activation not in act_map:
            raise ValueError(f"Invalid activation {activation}")

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
        self.mlp = MLP(
            in_dim,
            2,
            layer_widths=layer_widths or [256, 128, 64],
            activation=activation,
            batch_norm=batch_norm,
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


def train_discriminator(
    x_pos,
    x_neg,
    encoder,
    batch_size=128,
    max_epochs=100,
    lr=1e-3,
    weight_decay=1e-4,
    layer_widths=None,
    logger=None,
    deterministic=False,
    checkpoint_dir=None,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.backends.mps.is_available():
        device = "mps"

    X = torch.cat([x_pos, x_neg], dim=0)
    Y = torch.cat(
        [torch.ones(x_pos.shape[0], dtype=torch.long), torch.zeros(x_neg.shape[0], dtype=torch.long)],
        dim=0,
    )

    train_idx = int(0.8 * len(X))
    val_idx = int(0.9 * len(X))
    train_dataset = torch.utils.data.TensorDataset(X[:train_idx], Y[:train_idx])
    val_dataset = torch.utils.data.TensorDataset(X[train_idx:val_idx], Y[train_idx:val_idx])
    test_dataset = torch.utils.data.TensorDataset(X[val_idx:], Y[val_idx:])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = Discriminator(
        in_dim=x_pos.shape[1],
        lr=lr,
        weight_decay=weight_decay,
        layer_widths=layer_widths,
    )
    if checkpoint_dir is not None:
        os.makedirs(checkpoint_dir, exist_ok=True)
    callbacks = [
        pl.callbacks.EarlyStopping(monitor="val_loss", patience=20, mode="min"),
        pl.callbacks.ModelCheckpoint(
            monitor="val_loss",
            save_top_k=1,
            mode="min",
            filename="discriminator",
            dirpath=checkpoint_dir,
        ),
    ]
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        log_every_n_steps=10,
        accelerator=device,
        devices=1,
        callbacks=callbacks,
        logger=logger,
        deterministic=deterministic,
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    trainer.test(ckpt_path="best", dataloaders=test_loader)
    return Discriminator.load_from_checkpoint(callbacks[1].best_model_path)
