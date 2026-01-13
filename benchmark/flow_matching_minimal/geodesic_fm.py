import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_hidden_layers=3):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class SimpleCondCurve(nn.Module):
    """Lightweight conditional curve used for flow matching."""

    def __init__(self, input_dim, hidden_dim, num_layers=3, embed_t=False, scale_factor=1.0):
        super().__init__()
        self.embed_t = embed_t
        self.scale_factor = scale_factor
        feature_dim = input_dim * 2 + (hidden_dim if embed_t else 1)
        self.modifier = MLP(feature_dim, hidden_dim, input_dim, num_hidden_layers=num_layers)
        self.t_mlp = (
            MLP(1, hidden_dim, hidden_dim, num_hidden_layers=2) if embed_t else None
        )

    def forward(self, x0, x1, ts):
        ts = ts.unsqueeze(-1) if ts.dim() == 1 else ts  # [T, 1]
        num_steps, batch_size, dim = ts.size(0), x0.size(0), x0.size(1)
        t_rep = ts.view(num_steps, 1, 1).expand(-1, batch_size, 1)
        base_curve = x0.unsqueeze(0) + (x1 - x0).unsqueeze(0) * t_rep

        t_feat = self.t_mlp(t_rep.reshape(-1, 1)).view(
            num_steps, batch_size, -1
        ) if self.embed_t else t_rep
        if t_feat.device != x0.device:
            t_feat = t_feat.to(x0.device)
        features = torch.cat(
            [x0.expand(num_steps, -1, -1), x1.expand(num_steps, -1, -1), t_feat],
            dim=-1,
        )
        envelope = self.scale_factor * (1 - (t_rep * 2 - 1).pow(2))
        offsets = self.modifier(features.reshape(-1, features.size(-1))).reshape(
            num_steps, batch_size, dim
        )
        return base_curve + envelope * offsets


class GeodesicFlowMatching(pl.LightningModule):
    """Minimal reimplementation of the training loop used in the original notebook."""

    def __init__(
        self,
        func,
        encoder,
        input_dim,
        hidden_dim=64,
        scale_factor=1.0,
        embed_t=False,
        num_layers=3,
        n_tsteps=100,
        lr=1e-3,
        weight_decay=1e-4,
        flow_weight=1.0,
        length_weight=1.0,
    ):
        super().__init__()
        self.func = func if func is not None else (lambda x: x)
        self.encoder = encoder
        self.flow_model = MLP(input_dim + 1, hidden_dim, input_dim, num_hidden_layers=num_layers)
        self.cond_curve = SimpleCondCurve(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            embed_t=embed_t,
            scale_factor=scale_factor,
        )
        self.n_tsteps = n_tsteps
        self.lr = lr
        self.weight_decay = weight_decay
        self.flow_weight = flow_weight
        self.length_weight = length_weight
        self.save_hyperparameters()

    def _time_grid(self, device):
        return torch.linspace(0, 1, self.n_tsteps, device=device)

    def cc(self, x0, x1, ts):
        return self.cond_curve(x0, x1, ts)

    @staticmethod
    def _finite_diff_velocity(curve, ts):
        dt = ts[1:] - ts[:-1]
        vel = (curve[1:] - curve[:-1]) / dt.view(-1, 1, 1)
        # pad last step so shapes align with curve length
        return torch.cat([vel, vel[-1:].clone()], dim=0)

    def _flow_inputs(self, curve, ts):
        t_feat = ts.view(-1, 1, 1).expand(-1, curve.size(1), 1)
        return torch.cat([curve, t_feat], dim=-1)

    def step(self, batch, stage):
        x0, x1 = batch
        ts = self._time_grid(x0.device)
        curve = self.cc(x0, x1, ts)

        geom_curve_flat = self.func(curve.reshape(-1, curve.size(-1)))
        geom_curve = geom_curve_flat.view(ts.size(0), x0.size(0), -1)
        target_vel = self._finite_diff_velocity(curve, ts)
        length_loss = torch.norm(self._finite_diff_velocity(geom_curve, ts), dim=-1).mean()

        flow_pred = self.flow_model(
            self._flow_inputs(curve, ts).reshape(-1, curve.size(-1) + 1)
        ).reshape_as(curve)
        flow_loss = F.mse_loss(flow_pred, target_vel)

        loss = self.length_weight * length_loss + self.flow_weight * flow_loss
        self.log(f"{stage}_length", length_loss, prog_bar=True, on_epoch=True)
        self.log(f"{stage}_flow", flow_loss, prog_bar=True, on_epoch=True)
        self.log(f"{stage}_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self.step(batch, "test")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
