import numpy as np
import torch
import scipy
import wandb
from torch.utils.data import DataLoader, TensorDataset, Dataset, WeightedRandomSampler
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

import copy
import torch
from torch.utils.data import Dataset, Sampler, DataLoader
import pytorch_lightning as pl
import os
import pickle
import torch
from torch.utils.data import TensorDataset, Sampler, DataLoader
import pytorch_lightning as pl
from models.fm_models import *
from models.score_models import *
from models.metric_models import *
from models.embed_models import *
from utils.preprocess import *
from datasets.dataset import *
from utils.frozen import *
from utils.callback import *
from models.modules import *
from models.cfm import *
from models.cvae import *
from models.ema import *
from models.energy_models import *
from models.pita_models import *
from utils.summary_stat import *
from datasets.process import *

from benchmark.mfm.mfm_metric_models import *

import os
import sys
sys.path.append(os.path.abspath("conditional-flow-matching"))
    
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
import torch
import torchsde
from torchdyn.core import NeuralODE
from tqdm import tqdm

from torchcfm.conditional_flow_matching import *
from torchcfm.models import MLP
from torchcfm.utils import plot_trajectories, torch_wrapper

import gc

def build_metric(config):

    metric_net = SimpleScoreNet(input_dim=config.pc_dim,
                      output_dim=1,
                      hidden_dim=config.hidden_dim,
                      num_layers=config.num_layers)

    if config.mfm_benchmark:
        print("USING MFM")
        assert config.metric_max_epochs>5, "Need to train the metric tensor when you use MFM"
        metric_model = MetricNetMFM(metric_net=metric_net,
                                    K = config.K,
                                    kappa=config.kappa,
                                    config=config)
    else:
        metric_model = None #TODO

    return metric_model

def build_embed(config, adata, metric_model):

    sample_rescale = torch.from_numpy(adata.uns['std'])

    embed_net = SimpleEmbedNet(input_dim=config.pc_dim,
                  output_dim=config.latent_dim,
                  layer_norm=True,
                  hidden_dims=[config.hidden_dim]*config.num_layers,
                  sample_rescale=sample_rescale,
                  rescale=config.rescale,
                  skip=config.skip)

    geo_net = SinNet(input_dim=2 * config.pc_dim,
                      cond_dim=config.cond_dim,
                      output_dim=config.pc_dim,
                      num_freq=config.num_freq,
                      layer_norm=True,
                      hidden_dims=[config.hidden_dim]*config.num_layers,
                      rescale=config.rescale)

    flow_matcher = MetricFlowMatcher(sigma=config.sigma, geo_net = geo_net, embed_net = embed_net, no_ot = config.fast_ot)

    timepoints = sorted(adata.obs['timepoint'].unique().tolist())
    t_global_min, t_global_max = min(timepoints), max(timepoints)
    
    embed_model = EmbedNetTrainBase(flow_matcher=flow_matcher,
                                    metric_model=metric_model,
                                    geo_net=geo_net,
                                    embed_net=embed_net,
                                    config=config,
                                    t_global_min=t_global_min,
                                    t_global_max=t_global_max,
                                    sample_rescale=sample_rescale)

    return embed_model

def build_flow(config, adata, conditions, embed_model):

    flow_net = SinNet(input_dim=config.pc_dim,
                      cond_dim=config.cond_dim,
                      output_dim=config.pc_dim,
                      num_freq=config.num_freq,
                      layer_norm=True,
                      hidden_dims=[config.hidden_dim]*config.num_layers)


    timepoints = sorted(adata.obs['timepoint'].unique().tolist())
    t_global_min, t_global_max = min(timepoints), max(timepoints)
    sample_rescale = torch.from_numpy(adata.uns['std'])
    
    flow_model = MetricFlowNetTrainBase(flow_matcher=embed_model.flow_matcher,
                             flow_net=flow_net,
                             geo_net=embed_model.geo_net,
                             embed_net=embed_model.embed_net,
                             conditions=conditions,
                             config=config,
                             t_global_min=t_global_min,
                             t_global_max=t_global_max,
                             sample_rescale=sample_rescale)

    return flow_model

def build_trainer(config, wandb_logger, phase=None):
    callbacks = []
    if phase is None:
        max_epochs = config.max_epochs
    elif phase == "metric":
        max_epochs = config.metric_max_epochs
    elif phase == "embed":
        max_epochs = config.embed_max_epochs
        callbacks.append(DatasetUpdateCallback())
    elif phase == "flow":
        max_epochs = config.flow_max_epochs
        callbacks.append(DatasetUpdateCallback())
    trainer = pl.Trainer(
        accelerator="cpu" if config.force_cpu else "gpu", 
        logger=wandb_logger,
        callbacks = callbacks,
        log_every_n_steps=1, 
        max_epochs=max_epochs, 
        gradient_clip_val=config.gradient_clip_val,
        enable_checkpointing=False,
        enable_progress_bar=False,
        )
    return trainer


def run_full_model(config = None, project = None, adata = None, values = None, conditions = None, dataset = None):
    
    original_config = config.copy()
    X, y = extract_score_dataset(adata, values, n_neighbors=config.n_neighbors, resolution=config.resolution) #TODO: FIX

    classifier_model, metric_model, embed_model, flow_model = None, None, None, None

    phase_list =  ['classifier', 'metric', 'embed', 'flow']
    
    for i, phase in enumerate(phase_list):

        print(f"Running phase {phase}:.......")
        
        ### wandb ###
        wandb_logger = WandbLogger(project=project, name=phase, log_model=True)
        if original_config:
            wandb.init(config = original_config, project=project, reinit=True)
        else:
            wandb.init(reinit=True)
        config = wandb.config

        if phase == 'classifier':
            classifier_model = build_classifier(config)    

        if phase == 'metric':
            metric_model = build_metric(config, classifier_model)

        if phase == 'embed':
            embed_model = build_embed(config, adata, metric_model)

        if phase == 'flow':
            flow_model = build_flow(config, adata, embed_model)


        ### build dataset ###
        if phase in ['classifier', 'metric']:

            train_dataset = TensorDataset(X, y)
            train_dataloader = DataLoader(train_dataset, batch_size = config.score_batch_size)
            
        else:

            if config.fast_ot:
                #TODO: hard-coded?
                update_epoch_rate = 50 if phase == "embed" else 10000
                train_dataset = ShufflingOTDataset(dataset, config.flow_batch_size, conditions, update_epoch_rate)
            else:
                train_dataset = ShufflingDataset(dataset, config.flow_batch_size, conditions)
            train_dataloader = DataLoader(train_dataset, batch_size = config.loader_batch_size, shuffle=True)

        
        
        ### train ###
        trainer = build_trainer(config, wandb_logger, phase)
        model = {'metric': metric_model,
                 'embed': embed_model,
                 'flow': flow_model}[phase]
        wandb_logger.watch(model, log="all")
        trainer.fit(model=model, train_dataloaders=train_dataloader)
        wandb.finish()

        try:
            import wandb as _wandb
            _wandb.unwatch(model)
        except Exception:
            pass
        
        model.eval()
        
        ### cleanup ###

        if phase == "classifier":
            freeze_params(classifier_model.metric_net)

        if phase == "metric":
            freeze_params(metric_model.metric_net)
            
        if phase == "embed":
            freeze_params(embed_model.embed_net)

        if phase == "flow":
            pass

        model.to("cpu")

    return classifier_model, metric_model, embed_model, flow_model

