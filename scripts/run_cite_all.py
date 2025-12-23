import numpy as np
import torch
import scanpy as sc
import anndata as ad
import os
import pandas as pd
import sys

sys.path.append(os.path.abspath("."))
from utils.preprocess import *

os.environ["PYKEOPS_VERBOSE"] = "0"
sys.path.append(os.path.abspath("conditional-flow-matching"))

import matplotlib.pyplot as plt
import torchsde
from torchdyn.core import NeuralODE
from tqdm import tqdm

from torchcfm.conditional_flow_matching import *
from torchcfm.models import MLP
from visualize.plots import *

from models.modules import *
from visualize.plots import *

from scripts.run_model import *
from eval.eval import *
import copy


def run_model_once(config, adata_train, adata_raw, values, conditions, dataset,
                   project=None, t=2, num_traj=6999):
    # Train
    _, _, _, _, _, _, flow_model = run_full_model(
        config, project, adata_train, values, conditions, dataset
    )
    # Evaluate
    score = predict(flow_model, adata_raw, 'ctrl-inj', conditions, num_traj, t, p=1, steps=500)
    return score


def run_models_by_timepoint(config, adata, adata_raw, values, heldout_timepoints,
                            project="EB-benchmark", num_traj=6999):
    models_to_run = ['EGG-FM', 'straight_line', 'mfm']
    rows = []

    for t in heldout_timepoints:
        print(f"\n========== Held-out timepoint: {t} ==========")
        timepoints = sorted(adata.obs['timepoint'].unique())
        t_minus = timepoints[timepoints.index(t)-1]
        t_plus = timepoints[timepoints.index(t)+1]
        train_bool = adata.obs['timepoint'].isin([t_minus, t_plus])
        adata_train = adata[train_bool].copy()

        # Build dataset/conditions from training-only data
        conditions, dataset = extract_dataset(adata_train, values)

        row = {'heldout_timepoint': int(t)}

        for model_type in models_to_run:
            print(f"--- Running benchmark for: {model_type} ---")
            temp_config = copy.deepcopy(config)

            # Configure variants
            if model_type == 'straight_line':
                temp_config.rescale = 0.0
                temp_config.pita_steps = 1
                temp_config.score_max_epochs = 2
                temp_config.energy_max_epochs = 2
                temp_config.metric_max_epochs = 2
                temp_config.embed_max_epochs = 2
            elif model_type == 'mfm':
                temp_config.mfm_benchmark = True
                temp_config.metric_max_epochs = 2000
                temp_config.pita_steps = 1
                temp_config.score_max_epochs = 2
                temp_config.energy_max_epochs = 2
            # EGG-FM uses defaults

            score = run_model_once(
                temp_config, adata_train, adata_raw, values, conditions, dataset,
                project=f"{project}-{model_type}-t{t}", t=t, num_traj=num_traj
            )
            row[model_type] = score

        rows.append(row)

    # Assemble table: columns are model types; rows are held-out timepoints
    results = pd.DataFrame(rows).sort_values('heldout_timepoint')
    results.rename(columns={'straight_line': 'CFM', 'mfm': 'MFM'}, inplace=True)
    # Order columns nicely
    col_order = ['heldout_timepoint', 'EGG-FM', 'CFM', 'MFM']
    results = results[[c for c in col_order if c in results.columns]]
    return results

import argparse

if __name__ == "__main__":
    energy_only = False
    d = 5

    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    config = Config(
        {
            "model_class": "metricflow",
            "score_max_epochs": 500,
            "energy_max_epochs": 3000,
            "metric_max_epochs": 2,
            "embed_max_epochs": 2 if energy_only else 500,
            "flow_max_epochs": 2 if energy_only else 2000,
            # "score_max_epochs": 2,
            # "energy_max_epochs": 2,
            # "metric_max_epochs": 2,
            # "embed_max_epochs": 2 if energy_only else 2,
            # "flow_max_epochs": 2 if energy_only else 2,
            "lr": 1e-4,
            "dropout": 0.0,
            "pc_dim": d,
            "cond_dim": d,
            "hidden_dim": 512,
            "score_batch_size": 4096,
            "flow_batch_size": 512,
            "num_freq": 32,
            "num_layers": 4,
            "force_cpu": False,
            "gradient_clip_val": 10,
            "loader_batch_size": 20,
            "warmup_steps": 0,
            "ema_decay": 1-1e-3,

            "pita_steps": 2,
            "mfm_benchmark": False,
            "score_alpha": 1.0,

            "energy_noise_sigma": 0.0,
            "metric_scale": 1,
            "metric_sigma": 0.05,
            "metric_exponent": 1.0,

            "ot_in_embed": True,
            "fast_ot": False,

            "num_sigmas": 20,
            "sigma_min": 0.02,
            "sigma_max": 0.3,

            "sigma_dim": 32,

            "score_beta_min": 1.0,
            "score_beta_max": 1.0,

            "latent_dim": 100,

            "skip": True,
            "rescale": 0.5,

            "pre_low_q": .05,
            "pre_high_q": .95,
            "low_q": .05,
            "high_q": .95,

            "weight_beta": 0.2,
            "gamma": 0.5,
            "sigma": 0.2,

            "K": 2000,
            "kappa": 1.0,

            "n_neighbors": 10,
            "resolution": 0.3,
        }
    )

    # Load + normalize once
    # adata, values = process_data(pc_dim=config.pc_dim, data="cite")
    # adata.uns['std'] = np.std(adata.obsm['X_pca'], axis=0, keepdims=True)
    # adata_raw = adata.copy()
    # adata.obsm['X_pca'] /= adata.uns['std']

    adata, values = process_data(pc_dim=config.pc_dim, data="cite")
    adata.obsm['X_pca'] /= np.std(adata.obsm['X_pca'], axis=0, keepdims=True)
    adata.uns['std'] = np.ones((1,d), dtype=np.float32)
    adata_raw = adata.copy()

    # Define which timepoints to hold out
    heldout_timepoints = [3, 4]

    # Run experiments
    project = "CITE-benchmark"
    num_traj = 6999
    results = run_models_by_timepoint(
        config, adata, adata_raw, values, heldout_timepoints,
        project=project, num_traj=num_traj
    )

    # Save CSV (columns are model types; rows are different held-out timepoints)
    out_csv = 'CITE_benchmark_W1_by_heldout_timepoint.csv'
    results.to_csv(out_csv, index=False)
    print(f"\nSaved results to: {out_csv}\n")
    print(results)