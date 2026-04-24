#!/usr/bin/env python
"""
LineageOT benchmark on mouse blood, zebrafish CNS, and zebrafish PA data.

Uses the cell-type tree prior from utils/lineage.py instead of barcode-based
tree fitting.  Cells of the same cell type share a single cluster node
(placed exactly at time_1 in the tree), so LineageOT treats same-type cells
as having identical predicted ancestor states — analogous to treating each
cell type as an independent clone.

Evaluation mirrors run_sf2m.py:
  - Train on endpoint pair (t0, t_last).
  - Predict intermediate timepoints via coupling-based linear interpolation.
  - Report W1 (POT emd2, euclidean) in original PCA space.

Run from the finfm project root:
    python sf2m_run/run_lineageot.py --dataset mouse_blood
    python sf2m_run/run_lineageot.py --dataset zebrafish_cns
    python sf2m_run/run_lineageot.py --dataset zebrafish_pa
"""

import argparse
import csv
import os
import sys
import anndata
import networkx as nx
import numpy as np
import ot as pot
import pandas as pd
import scanpy as sc

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "LineageOT"
    ),
)

import lineageot
from utils.lineage import ZEBRAFISH_NEURAL_ADJACENCY, ARCH_NEURAL_ADJACENCY, run_paga_tree

MOUSE_BLOOD_CELLTYPES = [
    "Blood progenitors 1",
    "Blood progenitors 2",
    "Haematoendothelial progenitors",
    "Erythroid1",
    "Erythroid2",
    "Erythroid3",
]


# ── Tree construction ─────────────────────────────────────────────────────────


def build_celltype_tree(adata_late, time_late, time_early, cell_type_key="cell_type",
                        root_time_factor=1000):
    """
    Build a LineageOT-compatible DiGraph for cells at time_late.

    Each unique cell type present at time_late gets a cluster node placed
    exactly at time_early.  Because LineageOT's add_nodes_at_time() already
    finds a node at time_early, it does not split those edges, and all cells
    of the same type share that cluster node as their predicted ancestor.

    Cluster nodes connect via long edges to a shared root far in the past
    (large root_time_factor ≈ independent clones: negligible cross-type
    information sharing in the GP regression).

    Parameters
    ----------
    adata_late : AnnData
        Cells at time_late.  Rows must be in the same order as the slice
        adata[adata.obs[time_key] == time_late] that will be passed to
        fit_lineage_coupling, since leaf node i corresponds to row i.
    time_late, time_early : float
    cell_type_key : str
    root_time_factor : float
        Root placed at time_early - root_time_factor * (time_late - time_early).

    Returns
    -------
    networkx.DiGraph with node attributes 'time', 'time_to_parent' and
    edge attribute 'time' as required by LineageOT.
    """
    dt = time_late - time_early
    if dt <= 0:
        raise ValueError(f"time_late ({time_late}) must be > time_early ({time_early})")
    root_time = time_early - root_time_factor * dt

    T = nx.DiGraph()
    T.add_node("root", time=root_time, time_to_parent=0)

    cell_types = adata_late.obs[cell_type_key].astype(str).values
    unique_types = sorted(set(cell_types))

    # One cluster node per unique cell type, placed at time_early.
    for ct in unique_types:
        cnode = ("cluster", ct)
        T.add_node(cnode, time=time_early, time_to_parent=root_time_factor * dt)
        T.add_edge("root", cnode, time=root_time_factor * dt)

    # Leaf cells: integer indices 0..N-1 aligned with adata_late rows.
    for i, ct in enumerate(cell_types):
        cnode = ("cluster", ct)
        T.add_node(i, time=time_late, time_to_parent=dt)
        T.add_edge(cnode, i, time=dt)

    return T


# ── Data loading ──────────────────────────────────────────────────────────────


def _add_timepoint_col(adata):
    """Ensure adata.obs['timepoint'] (float) exists."""
    if "timepoint" in adata.obs:
        adata.obs["timepoint"] = adata.obs["timepoint"].astype(float)
        return
    if "stage" in adata.obs:
        stage_to_num = {
            "E6.5": 6.5, "E6.75": 6.75, "E7.0": 7.0, "E7.25": 7.25,
            "E7.5": 7.5, "E7.75": 7.75, "E8.0": 8.0, "E8.25": 8.25, "E8.5": 8.5,
        }
        adata.obs["timepoint"] = adata.obs["stage"].map(stage_to_num).astype(float)
        return
    if "day" in adata.obs:
        adata.obs["timepoint"] = adata.obs["day"].astype(float)
        return
    raise ValueError(
        "Cannot find time column: expected 'timepoint', 'stage', or 'day' in adata.obs."
    )


def _add_celltype_col(adata):
    """Ensure adata.obs['cell_type'] (str) exists."""
    if "cell_type" not in adata.obs:
        if "celltype" in adata.obs:
            adata.obs["cell_type"] = adata.obs["celltype"].astype(str)
        elif "cell_type_broad" in adata.obs:
            adata.obs["cell_type"] = adata.obs["cell_type_broad"].astype(str)
        else:
            raise ValueError("No cell type column found (tried 'cell_type', 'celltype', 'cell_type_broad').")
    else:
        adata.obs["cell_type"] = adata.obs["cell_type"].astype(str)


def _finalize_data(adata, pc_dim, t0_index, t1_index):
    """
    Slice the window [t0, t1], compute normalisation stats, and return a
    standard data bundle used by both datasets.

    Returns
    -------
    adata_window : AnnData (window only, X_pca computed)
    tps : sorted list of timepoints in window
    t0, t1 : float (endpoint times)
    mean : (1, pc_dim)
    std  : float
    X_by_tp : dict  tp -> normalised (n_cells, pc_dim) array
    """
    all_tps = sorted(float(v) for v in adata.obs["timepoint"].unique())
    t0 = all_tps[t0_index]
    t1 = all_tps[t1_index]

    mask = (adata.obs["timepoint"] >= t0) & (adata.obs["timepoint"] <= t1)
    adata_window = adata[mask].copy()
    tps = sorted(float(v) for v in adata_window.obs["timepoint"].unique())

    X_raw = {
        tp: adata_window[np.isclose(adata_window.obs["timepoint"].values, tp)].obsm["X_pca"][
            :, :pc_dim
        ].astype(np.float32)
        for tp in tps
    }

    train_raw = np.concatenate([X_raw[t0], X_raw[t1]], axis=0)
    mean = train_raw.mean(axis=0, keepdims=True)
    std = float(train_raw.std(axis=0).max() * np.sqrt(train_raw.shape[1]))
    std = std if std > 0 else 1.0

    X_by_tp = {tp: (x - mean) / std for tp, x in X_raw.items()}
    return adata_window, tps, t0, t1, mean, std, X_by_tp


def load_mouse_blood(adata_path, pc_dim, t0_index, t1_index, paga_threshold=0.1):
    """Load mouse blood data and compute PAGA cell-type adjacency."""
    print(f"Loading mouse blood data from {adata_path} ...")
    adata = sc.read(adata_path)

    _add_celltype_col(adata)
    _add_timepoint_col(adata)

    adata = adata[adata.obs["cell_type"].isin(MOUSE_BLOOD_CELLTYPES)].copy()
    adata.obs["cell_type"] = (
        adata.obs["cell_type"].astype("category").cat.remove_unused_categories()
    )

    sc.tl.pca(adata, n_comps=pc_dim, mask_var=None)

    root_node = "Haematoendothelial progenitors"
    print("  Running PAGA for cell-type tree ...")
    adj = run_paga_tree(adata, "cell_type", threshold=paga_threshold, root_node=root_node)
    print(f"  PAGA adjacency: {adj}")

    adata_window, tps, t0, t1, mean, std, X_by_tp = _finalize_data(
        adata, pc_dim, t0_index, t1_index
    )
    return adata_window, adj, tps, t0, t1, mean, std, X_by_tp


def load_zebrafish_cns(adata_path, pc_dim, t0_index, t1_index,
                       tissue="Central Nervous System"):
    """Load zebrafish CNS data with hard-coded ZEBRAFISH_NEURAL_ADJACENCY."""
    print(f"Loading zebrafish CNS data from {adata_path} ...")
    adata = sc.read(adata_path)

    _add_celltype_col(adata)
    _add_timepoint_col(adata)

    # Optional tissue filter
    if tissue is not None and "tissue" in adata.obs:
        adata = adata[adata.obs["tissue"] == tissue].copy()

    # Ctrl-injection filter (zebrafish perturbation atlas)
    if "gene_target" in adata.obs:
        adata = adata[adata.obs["gene_target"] == "ctrl-inj"].copy()

    # Restrict to cell types covered by the hard-coded adjacency
    adj = ZEBRAFISH_NEURAL_ADJACENCY
    all_adj_types = set(adj.keys())
    for v in adj.values():
        all_adj_types.update(v)
    adata = adata[adata.obs["cell_type"].isin(all_adj_types)].copy()
    adata.obs["cell_type"] = (
        adata.obs["cell_type"].astype("category").cat.remove_unused_categories()
    )

    sc.tl.pca(adata, n_comps=pc_dim, mask_var=None)

    adata_window, tps, t0, t1, mean, std, X_by_tp = _finalize_data(
        adata, pc_dim, t0_index, t1_index
    )
    return adata_window, adj, tps, t0, t1, mean, std, X_by_tp


def load_zebrafish_pa(adata_path, pc_dim, t0_index, t1_index):
    """Load zebrafish Pharyngeal Arch data with hard-coded ARCH_NEURAL_ADJACENCY."""
    print(f"Loading zebrafish PA data from {adata_path} ...")
    adata = sc.read(adata_path)

    _add_celltype_col(adata)
    _add_timepoint_col(adata)

    if "tissue" in adata.obs:
        adata = adata[adata.obs["tissue"] == "Pharyngeal Arch"].copy()

    if "gene_target" in adata.obs:
        adata = adata[adata.obs["gene_target"] == "ctrl-inj"].copy()

    adj = ARCH_NEURAL_ADJACENCY
    all_adj_types = set(adj.keys())
    for v in adj.values():
        all_adj_types.update(v)
    adata = adata[adata.obs["cell_type"].isin(all_adj_types)].copy()
    adata.obs["cell_type"] = (
        adata.obs["cell_type"].astype("category").cat.remove_unused_categories()
    )

    sc.tl.pca(adata, n_comps=pc_dim, mask_var=None)

    adata_window, tps, t0, t1, mean, std, X_by_tp = _finalize_data(
        adata, pc_dim, t0_index, t1_index
    )
    return adata_window, adj, tps, t0, t1, mean, std, X_by_tp


# ── Evaluation ────────────────────────────────────────────────────────────────


def coupling_interpolate(coupling_X, X_early, X_late, alpha):
    """
    Predict normalised distribution at fractional time alpha in [0,1].

    For each early cell i, compute the expected late position under the
    row-normalised coupling, then linearly blend with the early position.

    Parameters
    ----------
    coupling_X : (n_early, n_late)
    X_early    : (n_early, D)  normalised
    X_late     : (n_late,  D)  normalised
    alpha      : float in (0, 1)

    Returns
    -------
    X_pred : (n_early, D)  normalised predicted positions at alpha
    """
    row_sums = coupling_X.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    C_norm = coupling_X / row_sums
    expected_late = C_norm @ X_late
    return (1.0 - alpha) * X_early + alpha * expected_late


def evaluate_w1(coupling, X_by_tp, t0, t1, eval_timepoints, mean, std):
    """
    Evaluate W1 at held-out intermediate timepoints using coupling interpolation.

    The coupling gives a distribution over (t0, t1) cell pairs.  For each
    intermediate t_eval, we interpolate at alpha = (t_eval - t0) / (t1 - t0)
    and compute W1 vs the true distribution (in unnormalised PCA space).
    """
    X_early_norm = X_by_tp[t0]
    X_late_norm = X_by_tp[t1]

    scores = {}
    for t_eval in eval_timepoints:
        if t_eval not in X_by_tp:
            print(f"  skipping t={t_eval:g} (not in data)")
            continue
        alpha = (t_eval - t0) / (t1 - t0)
        X_pred_norm = coupling_interpolate(coupling.X, X_early_norm, X_late_norm, alpha)

        X_pred = X_pred_norm * std + mean
        X_true = X_by_tp[t_eval] * std + mean

        a = np.ones(len(X_pred)) / len(X_pred)
        b = np.ones(len(X_true)) / len(X_true)
        M = pot.dist(X_pred, X_true, metric="euclidean")
        w1 = float(pot.emd2(a, b, M))
        scores[t_eval] = w1
        print(f"  t={t_eval:g}  alpha={alpha:.3f}  W1={w1:.4f}")

    return scores


# ── Main ──────────────────────────────────────────────────────────────────────


def run_lineageot(args):
    np.random.seed(args.seed)

    # ── Load data ──
    if args.dataset == "mouse_blood":
        adata, _, tps, t0, t1, mean, std, X_by_tp = load_mouse_blood(
            args.adata_path, args.pc_dim, args.t0_index, args.t1_index,
            paga_threshold=args.paga_threshold,
        )
    elif args.dataset == "zebrafish_cns":
        adata, _, tps, t0, t1, mean, std, X_by_tp = load_zebrafish_cns(
            args.adata_path, args.pc_dim, args.t0_index, args.t1_index,
        )
    elif args.dataset == "zebrafish_pa":
        adata, _, tps, t0, t1, mean, std, X_by_tp = load_zebrafish_pa(
            args.adata_path, args.pc_dim, args.t0_index, args.t1_index,
        )
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    heldout = [tp for tp in tps if t0 < tp < t1]
    print(f"\nDataset        : {args.dataset}  seed={args.seed}")
    print(f"Timepoints     : {[f'{tp:g}' for tp in tps]}")
    print(f"Train endpoints: [{t0:g}, {t1:g}]   dim={args.pc_dim}")
    print(f"Held-out       : {[f'{tp:g}' for tp in heldout]}")
    unique_types = sorted(adata.obs["cell_type"].unique().tolist())
    print(f"Cell types     : {unique_types}")

    # ── Build tree for late cells ──
    # Cells at t1 in their AnnData row order (must match the order fit_lineage_coupling
    # will use when it slices adata_endpoints by timepoint == t1).
    late_mask = np.isclose(adata.obs["timepoint"].values, t1)
    adata_late = adata[late_mask].copy()
    adata_late.obs["cell_type"] = adata_late.obs["cell_type"].astype(str)

    print(f"\nBuilding cell-type tree  ({adata_late.n_obs} late cells, {len(unique_types)} types) ...")
    tree = build_celltype_tree(
        adata_late, t1, t0,
        cell_type_key="cell_type",
        root_time_factor=args.root_time_factor,
    )
    print(f"  Nodes: {tree.number_of_nodes()}  Edges: {tree.number_of_edges()}")

    # ── Build AnnData for the two endpoints with normalised PCA ──
    early_mask = np.isclose(adata.obs["timepoint"].values, t0)
    adata_early = adata[early_mask].copy()

    n_early = adata_early.n_obs
    n_late = adata_late.n_obs

    # Concatenate obs metadata (drop sparse .X to avoid dtype issues)
    obs_combined = pd.concat(
        [adata_early.obs.reset_index(drop=True),
         adata_late.obs.reset_index(drop=True)],
        ignore_index=True,
    )
    obs_combined.index = [f"c{i}" for i in range(n_early + n_late)]

    X_pca_norm = np.concatenate([X_by_tp[t0], X_by_tp[t1]], axis=0)  # (n_early+n_late, D)

    adata_endpoints = anndata.AnnData(obs=obs_combined)
    adata_endpoints.obsm["X_pca_norm"] = X_pca_norm
    adata_endpoints.obs["timepoint"] = obs_combined["timepoint"].astype(float).values

    # ── Fit LineageOT coupling ──
    print(f"\nFitting LineageOT coupling  t0={t0:g} → t1={t1:g} ...")
    coupling = lineageot.fit_lineage_coupling(
        adata_endpoints,
        time_1=t0,
        time_2=t1,
        lineage_tree_t2=tree,
        time_key="timepoint",
        state_key="X_pca_norm",
        epsilon=args.epsilon,
        normalize_cost=True,
    )
    print(f"  Coupling shape: {coupling.X.shape}")

    # ── Evaluate ──
    print(f"\nEvaluating W1 at held-out timepoints ...")
    scores = evaluate_w1(coupling, X_by_tp, t0, t1, heldout, mean, std)

    if not scores:
        print("  No intermediate timepoints available.")
    else:
        print("\n  timepoint   W1")
        print("  ---------   ------")
        for tp, w1 in sorted(scores.items()):
            print(f"  t={tp:<8g}  {w1:.4f}")

    # ── Save ──
    os.makedirs(args.save_dir, exist_ok=True)
    tsv_path = os.path.join(args.save_dir, "w1_scores.tsv")
    with open(tsv_path, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["timepoint", "w1"])
        for tp, w1 in sorted(scores.items()):
            writer.writerow([tp, w1])
    print(f"\nScores → {tsv_path}")
    return scores


def main():
    p = argparse.ArgumentParser(description="LineageOT benchmark (cell-type tree prior)")
    p.add_argument(
        "--dataset", required=True, choices=["mouse_blood", "zebrafish_cns", "zebrafish_pa"],
        help="Dataset to benchmark",
    )
    p.add_argument(
        "--adata_path", default=None,
        help="Path to h5ad file.  Defaults: atlas/mouse_preprocessed.h5ad or zebrafish_neural.h5ad",
    )
    p.add_argument("--pc_dim", type=int, default=100)
    p.add_argument("--t0_index", type=int, default=None,
                   help="Index of first timepoint (default: 1 for both datasets)")
    p.add_argument("--t1_index", type=int, default=None,
                   help="Index of last timepoint  (default: 7 for mouse_blood, 4 for zebrafish_cns)")
    p.add_argument("--epsilon", type=float, default=0.05,
                   help="Sinkhorn regularisation for LineageOT")
    p.add_argument("--root_time_factor", type=float, default=1000,
                   help="Root placed at time_early - factor*(time_late-time_early)")
    p.add_argument("--paga_threshold", type=float, default=0.1,
                   help="PAGA edge threshold (mouse only)")
    p.add_argument("--save_dir", default="sf2m_run/outputs")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed (LineageOT is deterministic; seed affects data subsampling only)")
    args = p.parse_args()

    # Dataset-specific defaults
    if args.adata_path is None:
        args.adata_path = (
            "atlas/mouse_preprocessed.h5ad"
            if args.dataset == "mouse_blood"
            else "zebrafish_neural.h5ad"
        )
    if args.t0_index is None:
        args.t0_index = 1
    if args.t1_index is None:
        args.t1_index = 7 if args.dataset == "mouse_blood" else 4
        # zebrafish_pa shares the same h5ad and timepoint structure as zebrafish_cns

    run_lineageot(args)


if __name__ == "__main__":
    main()
