"""MIOFlow benchmark: mirrors run_model.train() using MIOFlow's neural ODE.

Install MIOFlow before running:
    pip install git+https://github.com/KrishnaswamyLab/MIOFlow.git

Data setup matches the CFM baseline:
  - Same process_data() call, same t0/t1 training split
  - Trains on t0 and t1 cells only (no intermediate timepoints seen during training)
  - Same heldout protocol as run_model.train() (one sample held out at t_val)
  - Evaluates with ot_dist(p=1, library='pot') in PCA space
"""

try:
    from MIOFlow.ode import ODEF, NeuralODE
    from MIOFlow.train import training_regimen
    from MIOFlow.losses import OT_loss
except ImportError as exc:
    raise ImportError(
        "MIOFlow is required for this benchmark.\n"
        "Install with: pip install git+https://github.com/KrishnaswamyLab/MIOFlow.git"
    ) from exc

import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from datasets.process import process_data
from eval.eval import ot_dist

# ---------------------------------------------------------------------------
# MIOFlow training hyperparameters (hardcoded; see MIOFlow paper defaults)
# ---------------------------------------------------------------------------
_N_LOCAL_EPOCHS      = 100   # pre-training epochs  (local loss: t0 → t1)
_N_EPOCHS            = 100   # main training epochs (global loss)
_N_POST_LOCAL_EPOCHS = 100   # post-training epochs (local loss)
_N_BATCHES           = 20    # gradient steps per epoch
_SAMPLE_SIZE         = 256   # cells sampled per timepoint per batch
_USE_DENSITY_LOSS    = True  # density regularization (MIOFlow paper default)
_LAMBDA_DENSITY      = 1.0   # weight for density loss


# ---------------------------------------------------------------------------
# ODE function: time-conditioned MLP implementing the ODEF interface
# ---------------------------------------------------------------------------

class MIOFlowODEFunc(ODEF):
    """MLP vector field with time concatenation, wrapping MIOFlow's ODEF base."""

    def __init__(self, input_dim: int, hidden_dim: int = 256, n_layers: int = 3):
        super().__init__()
        self.condition_dim = 0  # required by MIOFlow training loop
        layers = [nn.Linear(input_dim + 1, hidden_dim), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers.append(nn.Linear(hidden_dim, input_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # t can be a scalar, shape (,), or shape (B,) — expand to (B, 1)
        if t.dim() == 0:
            t_exp = t.view(1, 1).expand(z.shape[0], 1)
        elif t.dim() == 1:
            t_exp = t.unsqueeze(1).expand(z.shape[0], 1)
        else:
            t_exp = t.expand(z.shape[0], 1)
        return self.net(torch.cat([z, t_exp.to(z)], dim=-1))


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _adata_to_mioflow_df(adata, t0: float, t1: float,
                         use_rep: str = 'X_pca') -> pd.DataFrame:
    """Convert AnnData to MIOFlow-style DataFrame.

    MIOFlow's sample() filters df[df['samples'] == group_value], so the
    'samples' column must hold the same float values used in groups=[t0_n, t1_n].
    We normalise timepoints to [0, 1] here.
    """
    X = adata.obsm[use_rep]
    df = pd.DataFrame(X, columns=[f'd{i+1}' for i in range(X.shape[1])])
    raw_tp = adata.obs['timepoint'].values
    df['samples'] = (raw_tp - t0) / (t1 - t0)   # map to [0, 1]
    return df


def _norm_time(t: float, t0: float, t1: float) -> float:
    return (t - t0) / (t1 - t0)


# ---------------------------------------------------------------------------
# Prediction via ODE integration
# ---------------------------------------------------------------------------

def predict_mioflow(
    model: NeuralODE,
    x0: torch.Tensor,
    t0: float,
    t: float,
    t1: float,
) -> torch.Tensor:
    """Integrate ODE from t0 to t starting at x0; returns predicted cells at t.

    Times are normalised to [0, 1] before integration.
    Returns shape (n_cells, pc_dim) on CPU.
    """
    t_start = _norm_time(t0, t0, t1)   # 0.0
    t_end   = _norm_time(t,  t0, t1)   # in (0, 1]
    times = torch.tensor([t_start, t_end], dtype=x0.dtype, device=x0.device)
    with torch.no_grad():
        traj = model(x0, t=times, return_whole_sequence=True)
    return traj[-1].cpu()   # (n_cells, pc_dim)


# ---------------------------------------------------------------------------
# Main entry point — same signature as run_model.train()
# ---------------------------------------------------------------------------

def train(config, project, wandb_logger=None):
    """Train MIOFlow and evaluate on heldout timepoints.

    Mirrors run_model.train() in data loading, train/val splits, and
    evaluation protocol.  Returns (model, w1_scores, w1_val_scores).
    """
    if config.seeded:
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)

    # ---- Data: identical to run_model.train() ----
    adata = process_data(
        pc_dim=config.pc_dim,
        t0_index=config.t0_index,
        t1_index=config.t1_index,
        data=config.data,
        use_paga=config.use_paga,
        paga_threshold=config.paga_threshold,
        tissue=config.tissue,
    )

    timepoints = sorted(adata.obs['timepoint'].unique().tolist())
    t0, t1 = timepoints[config.t0_index], timepoints[config.t1_index]

    # Restrict to the [t0, t1] window
    adata = adata[(adata.obs['timepoint'] >= t0) & (adata.obs['timepoint'] <= t1)]

    # Train on endpoints only — mirrors the CFM setup in run_model.py
    adata_train = adata[adata.obs['timepoint'].isin([t0, t1])]

    # ---- Convert to MIOFlow format ----
    # Timepoints are normalised to [0, 1] so the ODE integrates over a unit interval
    df_train = _adata_to_mioflow_df(adata_train, t0, t1)
    t0_norm, t1_norm = 0.0, 1.0
    groups = [t0_norm, t1_norm]

    # ---- Build model ----
    use_cuda = not config.force_cpu and torch.cuda.is_available()
    func = MIOFlowODEFunc(
        input_dim=config.pc_dim,
        hidden_dim=config.hidden_dim,   # reuse network width from existing config
        n_layers=config.num_layers,
    )
    model = NeuralODE(func)
    model.norm = []   # MIOFlow training loop expects this attribute
    if use_cuda:
        model = model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    # ---- Train ----
    print("Training MIOFlow neural ODE...")
    training_regimen(
        n_local_epochs=_N_LOCAL_EPOCHS,
        n_epochs=_N_EPOCHS,
        n_post_local_epochs=_N_POST_LOCAL_EPOCHS,
        exp_dir="",           # only used for plots; irrelevant when plot_every=None
        model=model,
        df=df_train,
        groups=groups,
        optimizer=optimizer,
        criterion=OT_loss(),  # OT loss — closest analogue to OT-CFM
        use_cuda=use_cuda,
        sample_size=(_SAMPLE_SIZE,),
        sample_with_replacement=False,
        use_density_loss=_USE_DENSITY_LOSS,
        lambda_density=_LAMBDA_DENSITY,
        n_batches=_N_BATCHES,
        plot_every=None,      # suppress all plotting
    )
    model.eval()

    # ---- Heldout protocol: identical to run_model.train() ----
    t_val = timepoints[config.t0_index + 1]
    sample_val = (
        adata[adata.obs['timepoint'] == t_val]
        .obs[config.sample_key].unique().tolist()[0]
    )
    subset_val = (
        (adata.obs['timepoint'] != t_val)
        | (adata.obs[config.sample_key] == sample_val)
    )
    adata_val = adata[subset_val]           # only sample_val at t_val
    adata     = adata[adata.obs[config.sample_key] != sample_val]  # exclude sample_val

    device = next(model.parameters()).device

    # w1_val_scores: heldout sample at t_val
    x0_val = torch.tensor(
        adata_val[adata_val.obs['timepoint'] == t0].obsm['X_pca']
    ).float().to(device)
    pred_val = predict_mioflow(model, x0_val, t0, t_val, t1)
    true_val = torch.tensor(
        adata_val[adata_val.obs['timepoint'] == t_val].obsm['X_pca']
    ).float()
    w1_val_scores = torch.tensor([ot_dist(pred_val, true_val, p=1, library='pot')])

    # w1_scores: all intermediate timepoints (excluding sample_val)
    w1_scores = []
    for index in range(config.t0_index + 1, config.t1_index):
        t = timepoints[index]
        x0 = torch.tensor(
            adata[adata.obs['timepoint'] == t0].obsm['X_pca']
        ).float().to(device)
        pred = predict_mioflow(model, x0, t0, t, t1)
        true = torch.tensor(
            adata[adata.obs['timepoint'] == t].obsm['X_pca']
        ).float()
        w1_scores.append(ot_dist(pred, true, p=1, library='pot'))
    w1_scores = torch.tensor(w1_scores)

    return model, w1_scores, w1_val_scores
