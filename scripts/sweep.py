import numpy as np
import torch
import wandb
import os
from torch.utils.data import DataLoader
import pytorch_lightning as pl

import os 
import sys
sys.path.append(os.path.abspath("conditional-flow-matching"))
sys.path.append(os.path.abspath("."))
from pytorch_lightning.loggers import WandbLogger

from scripts.run_model import *
from eval.eval import *
import time

project = "cite-sweep"

torch.manual_seed(42)
np.random.seed(42)

d = 5
t = 4
normalize = True

from collections import namedtuple
Holdout = namedtuple('Holdout', ['t', 'gene'])
holdout = Holdout(t, 'ctrl-inj')

adata, values = process_data(pc_dim=d, data="cite")

#NORMALIZING::::

adata.uns['std'] = np.ones((1,d))
adata_raw = adata.copy()

if normalize:
    adata.uns['std'] = np.std(adata.obsm['X_pca'], axis=0, keepdims=True)
    adata.obsm['X_pca'] /= adata.uns['std']

timepoints = sorted(adata.obs['timepoint'].unique())
t_minus = timepoints[timepoints.index(holdout.t)-1]
t_plus = timepoints[timepoints.index(holdout.t)+1]

test_bool = (adata.obs['timepoint'] == holdout.t) & (adata.obs['gene_target'] == holdout.gene)
train_bool = adata.obs['timepoint'].isin([t_minus, t_plus])
adata_train = adata[train_bool]
adata_test = adata[test_bool]

conditions, dataset = extract_dataset(adata_train, values)







config = {        
    "model_class": "metricflow",
    "score_max_epochs": 3000,
    "energy_max_epochs": 5000,
    "metric_max_epochs": 2,
    "embed_max_epochs": 10000,
    "flow_max_epochs": 10000,

    # "score_max_epochs": 2,
    # "energy_max_epochs": 2,
    # "metric_max_epochs": 2,
    # "embed_max_epochs": 2,
    # "flow_max_epochs": 2,
    
    "lr": 1e-4,
    "dropout": 0.0,
    "pc_dim": d,
    "cond_dim": d,
    "hidden_dim": 512,
    "score_batch_size": 4096,
    "flow_batch_size": 256,
    "num_freq": 32,
    "num_layers": 5,
    "control_only": True,
    "force_cpu": False,
    "constrain": False,
    "gradient_clip_val": 10.0,
    "loader_batch_size": 20,
    "accumulate_grad_batches": 1,
    "warmup_steps": 0,
    "ema_decay": .999,

    "mfm_benchmark": False,

    "pita_steps": 2,

    "score_alpha": 1.0,

    "energy_noise_sigma": 0.0,
    "metric_scale": 10,
    "metric_sigma": 0.05,

    "ot_in_embed": True,
    "fast_ot": False,

    "num_sigmas": 20,
    "sigma_min": 0.01,
    "sigma_max": 0.2,

    "sigma_dim": 32,

    "score_beta_min": 1.0,
    "score_beta_max": 1.0,
    
    "latent_dim": 100,

    "skip": True,
    "rescale": 0.5,

    "pre_low_q": .05,
    "pre_high_q": .98,
    "low_q": .05,
    "high_q": .95,

    "weight_beta": 0.3,

    "gamma": 0.2,

    "sigma": 0.1,

    "n_neighbors": 10,
    "resolution": 0.3,
    
}





def run_model_sweep(config, project, adata, adata_raw, values, conditions, dataset):
    run = wandb.init(project=project, config=config)      
    cfg = run.config                      

    pre_score_models, pre_energy_models, score_model, energy_model, metric_model, embed_model, flow_model = run_full_model_for_sweeping(
        config=cfg,
        run=run,           
        adata=adata,
        values=values,
        conditions=conditions,
        dataset=dataset,
    )
    with torch.no_grad():
        flow_model.eval()
        one_wasserstein = predict(
            flow_model, adata_raw, value='ctrl-inj', conditions=conditions,
            num_traj=6999, t=t, p=1
        )
    run.log({"one_wasserstein": one_wasserstein})
    run.finish()
    return


sweep_config = {
    'method': 'random' # random, grid, bayesian
    }
metric = {
    'name': 'one_wasserstein',
    'goal': 'minimize'   
    }
sweep_config['metric'] = metric

parameters_dict = {}
for key in config:
    dic = {'values': [config[key]]}
    parameters_dict[key] = dic

parameters_dict['sigma_min'] = {'values': [0.01, 0.02, 0.05]}
parameters_dict['sigma_max'] = {'values': [0.1, 0.2, 0.3]}
parameters_dict['pre_low_q'] = {'values': [0.02, 0.05, 0.1]}
parameters_dict['pre_high_q'] = {'values': [0.98, 0.95, 0.9]}
parameters_dict['low_q'] = {'values': [0.02, 0.05, 0.1]}
parameters_dict['high_q'] = {'values': [0.98, 0.95, 0.9]}
parameters_dict['weight_beta'] = {'values': [0.3, 0.5, 0.9]}
parameters_dict['gamma'] = {'values': [0.5, 2.0]}
parameters_dict['sigma'] = {'values': [0.2]}
parameters_dict['num_layers'] = {'values': [4]}
parameters_dict['metric_scale'] = {'values': [2.0, 5.0, 10.0]}

sweep_config['parameters'] = parameters_dict


########################################

if __name__ == "__main__":

    sweep_id = wandb.sweep(sweep=sweep_config, project=project)
    wandb.agent(sweep_id, function=lambda: run_model_sweep(config, project, adata_train, adata_raw, values, conditions, dataset))