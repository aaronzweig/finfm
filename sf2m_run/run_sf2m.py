#!/usr/bin/env python
"""
SF2M on zebrafish neural data.

Training loop mirrors single-cell_example.ipynb exactly.
W1 evaluation uses POT (ot.emd2, euclidean) in normalized PCA space.

Run from the finfm project root:
    python sf2m_run/run_sf2m.py --adata_path zebrafish_neural.h5ad
"""

import argparse
import csv
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torchsde
from torchdyn.core import NeuralODE
from tqdm import tqdm
import ot as pot

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torchcfm.conditional_flow_matching import SchrodingerBridgeConditionalFlowMatcher

MOUSE_BLOOD_CELLTYPES = [
    "Blood progenitors 1",
    "Blood progenitors 2",
    "Haematoendothelial progenitors",
    "Erythroid1",
    "Erythroid2",
    "Erythroid3",
]
MOUSE_BRAIN_CELLTYPES = [
    "Caudal epiblast",
    "NMP",
    "Caudal neurectoderm",
    "Spinal cord",
]


# ═══════════════════════════════════════════════════════════════════════════════
# MLP  (same as single-cell_example.ipynb)
# ═══════════════════════════════════════════════════════════════════════════════

class MLP(nn.Module):
    """
    4-layer SELU MLP.  When time_varying=True, forward() expects the input
    already concatenated as [x, t], so effective input dim = dim + 1.
    """
    def __init__(self, dim, out_dim=None, w=64, time_varying=False):
        super().__init__()
        if out_dim is None:
            out_dim = dim
        in_dim = dim + 1 if time_varying else dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, w), nn.SELU(),
            nn.Linear(w, w),      nn.SELU(),
            nn.Linear(w, w),      nn.SELU(),
            nn.Linear(w, out_dim),
        )

    def forward(self, x):
        return self.net(x)


# ── ODE / SDE wrappers ────────────────────────────────────────────────────────

class TorchWrapper(nn.Module):
    """Adapts MLP(cat[x,t]) to torchdyn's f(t, x) interface."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, t, x, **kwargs):
        t_b = t.reshape(1).expand(x.shape[0])   # (N,)
        return self.model(torch.cat([x, t_b[:, None]], dim=-1))


class SF2M_SDE(nn.Module):
    noise_type = "diagonal"
    sde_type   = "ito"

    def __init__(self, drift, score, dim, sigma):
        super().__init__()
        self.drift = drift
        self.score = score
        self.dim   = dim
        self.sigma = sigma

    def f(self, t, y):
        t_b = t.reshape(1).expand(y.shape[0])
        x   = torch.cat([y, t_b[:, None]], dim=-1)
        return self.drift(x) + self.score(x)

    def g(self, t, y):
        return torch.ones_like(y) * self.sigma


# ═══════════════════════════════════════════════════════════════════════════════
# Data
# ═══════════════════════════════════════════════════════════════════════════════

def load_data(adata_path, pc_dim, t0_index, t1_index, tissue=None, mouse_subset="none"):
    import scanpy as sc
    adata = sc.read(adata_path)

    # Normalize naming across datasets.
    if "cell_type" not in adata.obs:
        if "celltype" in adata.obs:
            adata.obs["cell_type"] = adata.obs["celltype"].astype(str)
        elif "cell_type_broad" in adata.obs:
            adata.obs["cell_type"] = adata.obs["cell_type_broad"].astype(str)

    # Add numeric timepoint when missing (mouse_preprocessed.h5ad has 'stage').
    if "timepoint" not in adata.obs:
        if "stage" in adata.obs:
            stage_to_num = {
                "E6.5": 6.5,
                "E6.75": 6.75,
                "E7.0": 7.0,
                "E7.25": 7.25,
                "E7.5": 7.5,
                "E7.75": 7.75,
                "E8.0": 8.0,
                "E8.25": 8.25,
                "E8.5": 8.5,
            }
            adata.obs["timepoint"] = adata.obs["stage"].map(stage_to_num).astype(float)
        elif "day" in adata.obs:
            adata.obs["timepoint"] = adata.obs["day"].astype(float)
        else:
            raise ValueError(
                "Could not find time annotations. Expected one of: "
                "obs['timepoint'], obs['stage'], obs['day']."
            )

    if adata.obs["timepoint"].isna().any():
        missing = int(adata.obs["timepoint"].isna().sum())
        raise ValueError(f"Found {missing} cells with missing numeric timepoint values.")

    if mouse_subset != "none":
        if "cell_type" not in adata.obs:
            raise ValueError("mouse_subset requested, but no cell type column found in adata.obs.")
        if mouse_subset == "blood":
            keep = MOUSE_BLOOD_CELLTYPES
        elif mouse_subset == "brain":
            keep = MOUSE_BRAIN_CELLTYPES
        else:
            raise ValueError(f"Unknown mouse_subset='{mouse_subset}'.")
        adata = adata[adata.obs["cell_type"].isin(keep)].copy()
        if adata.n_obs == 0:
            raise ValueError(f"No cells left after mouse_subset='{mouse_subset}'.")

    # Match datasets/process.py behavior for zebrafish tissue subsetting.
    if tissue is not None:
        if "tissue" not in adata.obs:
            raise ValueError(
                f"Requested tissue='{tissue}', but 'tissue' is missing in adata.obs."
            )
        available_tissues = sorted(set(str(t) for t in adata.obs["tissue"].tolist()))
        adata = adata[adata.obs["tissue"] == tissue].copy()
        if adata.n_obs == 0:
            raise ValueError(
                f"No cells found for tissue='{tissue}'. "
                f"Available tissues: {available_tissues}"
            )
        # Same as process.py: recompute PCA on the tissue subset.
        sc.tl.pca(adata, n_comps=pc_dim, mask_var=None)
    elif "X_pca" not in adata.obsm or adata.obsm["X_pca"].shape[1] < pc_dim:
        # Fallback for adata files that do not already include enough PCs.
        sc.tl.pca(adata, n_comps=pc_dim, mask_var=None)

    all_tps = sorted(float(tp) for tp in adata.obs["timepoint"].unique().tolist())
    t0, t1  = all_tps[t0_index], all_tps[t1_index]
    adata   = adata[(adata.obs["timepoint"] >= t0) & (adata.obs["timepoint"] <= t1)]
    tps     = sorted(float(tp) for tp in adata.obs["timepoint"].unique().tolist())

    # One array per timepoint in the selected window.
    X_raw = {
        tp: adata[adata.obs["timepoint"] == tp].obsm["X_pca"][:, :pc_dim].astype(np.float32)
        for tp in tps
    }

    # Normalize using training endpoints only, so intermediate times remain held out.
    train_raw = np.concatenate([X_raw[t0], X_raw[t1]], axis=0)
    mean = train_raw.mean(axis=0, keepdims=True)  # (1, D)
    std = float(train_raw.std(axis=0).max() * np.sqrt(train_raw.shape[1]))
    if std == 0.0:
        std = 1.0

    X_by_tp = {tp: (x - mean) / std for tp, x in X_raw.items()}
    X_train = [X_by_tp[t0], X_by_tp[t1]]

    return X_train, X_by_tp, t0, t1, tps, mean, std


# ═══════════════════════════════════════════════════════════════════════════════
# get_batch  — exact copy from single-cell_example.ipynb
# ═══════════════════════════════════════════════════════════════════════════════

def get_batch(FM, X, batch_size, n_times, device, return_noise=False):
    """Construct a batch with points from each consecutive timepoint pair."""
    ts     = []
    xts    = []
    uts    = []
    noises = []
    for t_start in range(n_times - 1):
        x0 = (
            torch.from_numpy(X[t_start][np.random.randint(X[t_start].shape[0], size=batch_size)])
            .float().to(device)
        )
        x1 = (
            torch.from_numpy(X[t_start + 1][np.random.randint(X[t_start + 1].shape[0], size=batch_size)])
            .float().to(device)
        )
        if return_noise:
            t, xt, ut, eps = FM.sample_location_and_conditional_flow(x0, x1, return_noise=True)
            noises.append(eps)
        else:
            t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)
        ts.append(t + t_start)
        xts.append(xt)
        uts.append(ut)

    t  = torch.cat(ts)
    xt = torch.cat(xts)
    ut = torch.cat(uts)
    if return_noise:
        return t, xt, ut, torch.cat(noises)
    return t, xt, ut


# ═══════════════════════════════════════════════════════════════════════════════
# Training  — exact loop from single-cell_example.ipynb
# ═══════════════════════════════════════════════════════════════════════════════

def train(X, args, device):
    dim    = X[0].shape[1]
    sigma  = args.sigma
    n_times = len(X)

    sf2m_model       = MLP(dim=dim, time_varying=True, w=args.w).to(device)
    sf2m_score_model = MLP(dim=dim, time_varying=True, w=args.w).to(device)
    sf2m_optimizer   = torch.optim.AdamW(
        list(sf2m_model.parameters()) + list(sf2m_score_model.parameters()), 1e-4
    )
    SF2M = SchrodingerBridgeConditionalFlowMatcher(sigma=sigma)

    for i in tqdm(range(args.n_iters)):
        sf2m_optimizer.zero_grad()
        t, xt, ut, eps = get_batch(SF2M, X, args.batch_size, n_times, device, return_noise=True)
        lambda_t = SF2M.compute_lambda(t % 1)
        vt = sf2m_model(torch.cat([xt, t[:, None]], dim=-1))
        st = sf2m_score_model(torch.cat([xt, t[:, None]], dim=-1))
        flow_loss  = torch.mean((vt - ut) ** 2)
        score_loss = torch.mean((lambda_t[:, None] * st + eps) ** 2)
        if i % 1000 == 0:
            print(f"{i}: {flow_loss.item():.2f}, {score_loss.item():.2f}")
        loss = flow_loss + score_loss
        loss.backward()
        sf2m_optimizer.step()

    return sf2m_model, sf2m_score_model


# ═══════════════════════════════════════════════════════════════════════════════
# W1 evaluation  — POT ot.emd2 in normalized space (as in zebrafish notebook)
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_w1(
    sf2m_model,
    X_by_tp,
    t0,
    t1,
    eval_timepoints,
    mean,
    std,
    device,
    num_traj=2000,
    ode_steps=400,
):
    """
    Sample a full ODE trajectory from t0->t1 (normalized to [0, 1]), then
    evaluate W1 at requested unnormalized times against held-out data.
    """
    if t1 <= t0:
        raise ValueError(f"Invalid interval: t0={t0}, t1={t1}")

    t_span = torch.linspace(0.0, 1.0, ode_steps).to(device)

    node = NeuralODE(TorchWrapper(sf2m_model), solver="euler", sensitivity="adjoint")
    x0_all = X_by_tp[t0]
    n0 = min(num_traj, x0_all.shape[0])
    x0 = torch.from_numpy(x0_all[:n0]).float().to(device)
    with torch.no_grad():
        traj = node.trajectory(x0, t_span=t_span).cpu().numpy()  # (T, N, D) normalized

    scores = {}
    for t_eval in eval_timepoints:
        if t_eval <= t0 or t_eval >= t1:
            print(f"    skipping t={t_eval:g} (outside [{t0:g}, {t1:g}])")
            continue
        if t_eval not in X_by_tp:
            print(f"    skipping t={t_eval:g} (not found in data)")
            continue

        frac = (t_eval - t0) / (t1 - t0)  # normalize unnormalized time into [0, 1]
        step = int(round(frac * (len(t_span) - 1)))
        x_pred = traj[step] * std + mean   # (num_traj, D)
        x_true = X_by_tp[t_eval] * std + mean   # (n_cells, D)

        a = np.ones(len(x_pred)) / len(x_pred)
        b = np.ones(len(x_true)) / len(x_true)
        M = pot.dist(x_pred, x_true, metric="euclidean")
        w1 = float(pot.emd2(a, b, M))
        scores[t_eval] = w1
        print(f"    t={t_eval:g}  tau={frac:.3f}  step={step}  W1={w1:.4f}")

    return scores


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description="SF2M")
    p.add_argument("--adata_path",  default="zebrafish_neural.h5ad")
    p.add_argument(
        "--tissue",
        default=None,
        help="Optional tissue subset before time slicing.",
    )
    p.add_argument(
        "--mouse_subset",
        default="none",
        choices=["none", "blood", "brain"],
        help="Optional mouse subset by cell type (none|blood|brain).",
    )
    p.add_argument("--pc_dim",      type=int,   default=100)
    p.add_argument("--t0_index",    type=int,   default=1)
    p.add_argument("--t1_index",    type=int,   default=4)
    p.add_argument("--sigma",       type=float, default=0.25)
    p.add_argument("--n_iters",     type=int,   default=10000)
    p.add_argument("--batch_size",  type=int,   default=256)
    p.add_argument("--w",           type=int,   default=64,
                   help="MLP hidden width (notebook default: 64)")
    p.add_argument("--save_dir",    default="sf2m_run/outputs")
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument(
        "--eval_times",
        default="",
        help="Comma-separated unnormalized times to evaluate. Empty means all held-out times.",
    )
    p.add_argument("--num_traj",    type=int,   default=2000)
    p.add_argument("--ode_steps",   type=int,   default=400)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device     : {device}")

    print("Loading data...")
    if args.tissue is not None:
        print(f"Tissue subset        : {args.tissue}")
    if args.mouse_subset != "none":
        print(f"Mouse subset         : {args.mouse_subset}")
    X_train, X_by_tp, t0, t1, tps, mean, std = load_data(
        args.adata_path,
        args.pc_dim,
        args.t0_index,
        args.t1_index,
        tissue=args.tissue,
        mouse_subset=args.mouse_subset,
    )
    heldout_tps = [tp for tp in tps if t0 < tp < t1]
    print(f"Timepoints in window : {tps}")
    print(f"Train endpoints      : [{t0:g}, {t1:g}]   dim={X_train[0].shape[1]}")
    print(f"Held-out candidates  : {[f'{tp:g}' for tp in heldout_tps]}")

    print("Training SF2M...")
    sf2m_model, sf2m_score_model = train(X_train, args, device)

    os.makedirs(args.save_dir, exist_ok=True)
    ckpt = os.path.join(args.save_dir, "sf2m.pt")
    torch.save({
        "sf2m_model":       sf2m_model.state_dict(),
        "sf2m_score_model": sf2m_score_model.state_dict(),
        "mean": mean, "std": std, "timepoints": tps, "args": vars(args),
    }, ckpt)
    print(f"Checkpoint → {ckpt}")

    if args.eval_times.strip():
        eval_timepoints = [float(t.strip()) for t in args.eval_times.split(",") if t.strip()]
    else:
        eval_timepoints = heldout_tps

    print("\nEvaluating held-out W1 from full ODE trajectory...")
    print(f"  eval_times (unnormalized): {[f'{tp:g}' for tp in eval_timepoints]}")
    scores = evaluate_w1(
        sf2m_model,
        X_by_tp,
        t0,
        t1,
        eval_timepoints,
        mean,
        std,
        device,
        num_traj=args.num_traj,
        ode_steps=args.ode_steps,
    )
    print("\n  timepoint   W1")
    print("  ---------   ------")
    for tp, w1 in scores.items():
        print(f"  t={tp:<8}  {w1:.4f}")

    tsv_path = os.path.join(args.save_dir, "w1_scores.tsv")
    with open(tsv_path, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["timepoint", "w1"])
        for tp, w1 in scores.items():
            writer.writerow([tp, w1])
    print(f"\nScores → {tsv_path}")


if __name__ == "__main__":
    main()
