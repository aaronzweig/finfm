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



def build_score(config, conditions, pre_score_model):
    score_net = SimpleScoreNet(input_dim=config.pc_dim,
                      output_dim=config.pc_dim,
                      hidden_dim=config.hidden_dim,
                      num_layers=config.num_layers)

    if config.ema_decay is not None:
        score_net = EMA(score_net, config.ema_decay)

    score_model = ScoreNetTrainBaseAnneal(score_net=score_net,
                                          conditions=conditions,
                                          pre_score_model=pre_score_model,
                                          config=config)

    return score_model
    
def build_energy(config, conditions, score_model):

    energy_net = SimpleScoreNet(input_dim=config.pc_dim,
                      output_dim=config.pc_dim,
                      hidden_dim=config.hidden_dim,
                      num_layers=config.num_layers)

    energy_model = EnergyNetTrainBase(score_model=score_model,
                                      energy_net=energy_net,
                                      conditions=conditions,
                                      config=config)

    return energy_model

def build_score_pita(config, conditions, pre_score_model):
    print("score_pita")
    score_net = SimpleScoreNet(input_dim=config.pc_dim+2*config.sigma_dim,
                      output_dim=config.pc_dim,
                      hidden_dim=config.hidden_dim,
                      num_layers=config.num_layers)

    if config.ema_decay is not None:
        score_net = EMA(score_net, config.ema_decay)

    score_model = ScoreNetTrainPita(score_net=score_net,
                                          conditions=conditions,
                                          pre_score_model=pre_score_model,
                                          config=config)

    return score_model
    
def build_energy_pita(config, conditions, score_model):
    print("energy_pita")

    energy_net = SimpleScoreNet(input_dim=config.pc_dim+2*config.sigma_dim,
                      output_dim=config.pc_dim,
                      hidden_dim=config.hidden_dim,
                      num_layers=config.num_layers)

    # if config.ema_decay is not None:
    #     energy_net = EMA(energy_net, config.ema_decay)

    energy_model = EnergyNetTrainPita(score_model=score_model,
                                      energy_net=energy_net,
                                      conditions=conditions,
                                      config=config)

    return energy_model

def build_metric(config, conditions, energy_model):

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
                                    conditions=conditions,
                                    config=config)
    else:
        metric_model = MetricNetTrainBase(metric_net=metric_net,
                                          energy_model=energy_model,
                                          conditions=conditions,
                                          config=config)

    return metric_model

def build_embed(config, adata, conditions, metric_model):

    sample_rescale = torch.from_numpy(adata.uns['std'])
    
    # embed_net = SimpleDenseNet(input_dim=config.pc_dim,
    #               output_dim=config.latent_dim,
    #               layer_norm=True,
    #               hidden_dims=[config.hidden_dim]*config.num_layers,
    #               rescale=config.rescale,
    #               skip=config.skip)
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
                                    conditions=conditions,
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
    elif phase == "score":
        max_epochs = config.score_max_epochs
    elif phase == "energy":
        max_epochs = config.energy_max_epochs
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
    _, score_dataset, y = extract_score_dataset(adata, values, n_neighbors=config.n_neighbors, resolution=config.resolution)

    pre_score_model, pre_energy_model = None, None
    score_model, energy_model, metric_model, embed_model, flow_model = None, None, None, None, None
    pre_score_models, pre_energy_models = [], []

    phase_list =  ['score', 'energy'] * config.pita_steps + ['metric', 'embed', 'flow']
    weights = [1] * len(score_dataset)
    
    for i, phase in enumerate(phase_list):

        print(f"Running phase {phase}:.......")
        
        
        ### wandb ###
        wandb_logger = WandbLogger(project=project, name=phase, log_model=True)
        if original_config:
            wandb.init(config = original_config, project=project, reinit=True)
        else:
            wandb.init(reinit=True)
        config = wandb.config

        
        ### build models ###
        if phase == 'score':
            score_model = build_score_pita(config, conditions, pre_score_model)
            
            j = (i // 2)

            if j == config.pita_steps-1:
                #Final renormalization step, learn score over everything
                y *= 0

            # betas = np.linspace(config.score_beta_min, config.score_beta_max, config.pita_steps).tolist()
            betas = np.exp(np.linspace(np.log(config.score_beta_min),
                                       np.log(config.score_beta_max),
                                       config.pita_steps)).tolist()
            betas.reverse()
            score_model.score_beta = betas[j]

        if phase == 'energy':
            energy_model = build_energy_pita(config, conditions, score_model)

        if phase == 'metric':
            metric_model = build_metric(config, conditions, energy_model)
            metric_model.low_quantile, metric_model.high_quantile = metric_model_low_quantile, metric_model_high_quantile
            if config.mfm_benchmark:
                metric_model.train_dataloader = metric_model_train_dataloader

        if phase == 'embed':
            embed_model = build_embed(config, adata, conditions, metric_model)

        if phase == 'flow':
            flow_model = build_flow(config, adata, conditions, embed_model)


        ### build dataset ###
        if phase in ['score', 'energy', 'metric']:

            sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
            train_dataset = TensorDataset(score_dataset, y)
            train_dataloader = DataLoader(train_dataset, batch_size = config.score_batch_size, sampler=sampler, drop_last=True)
            
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
        model = {'score': score_model,
                 'energy': energy_model,
                 'metric': metric_model,
                 'embed': embed_model,
                 'flow': flow_model}[phase]
        wandb_logger.watch(model, log="all")
        # wandb_logger.watch(model, log="gradients", log_freq=200)  # no parameters hist, no graph
        trainer.fit(model=model, train_dataloaders=train_dataloader)
        wandb.finish()

        try:
            import wandb as _wandb
            _wandb.unwatch(model)
        except Exception:
            pass
        
        # Drop trainer & dataloader references that may keep CUDA tensors alive
        del trainer, train_dataloader
        try:
            del sampler
        except NameError:
            pass
        del wandb_logger
        
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        model.eval()
        
        ### cleanup ###
        if phase == "score":
            freeze_params(score_model.score_net)

        if phase == "energy":
            freeze_params(energy_model.energy_net)
            energy_model.score_model = None #Don't need score anymore
            
            train_dataloader = DataLoader(train_dataset, batch_size = config.score_batch_size, shuffle=False) #need consistent order

            ###
            if config.mfm_benchmark:
                metric_model_train_dataloader = train_dataloader #only necessary for MFM
            ###

            
            sigma_scalar = np.median(energy_model.energy_sigma)
            weights = get_new_weights(energy_model,
                                      train_dataloader,
                                      sigma_scalar=sigma_scalar,
                                      low=config.pre_low_q,
                                      high=config.pre_high_q,
                                      weight_beta=config.weight_beta)
            # pre_low_quantile, pre_high_quantile = get_energy_statistics_global(energy_model, 
            #                                                                    train_dataloader,
            #                                                                    low = config.pre_low_q,
            #                                                                    high = config.pre_high_q,
            #                                                                    sigma_scalar=sigma_scalar)
            
            # weights = get_energy_weights(energy_model, train_dataloader, sigma_scalar=sigma_scalar)
            # weights = torch.clamp(weights, min=pre_low_quantile, max=pre_high_quantile) - pre_high_quantile
            # weights = torch.exp(config.weight_beta * weights)


            
            metric_model_low_quantile, metric_model_high_quantile = get_energy_statistics_global(energy_model,
                                                                                                 train_dataloader,
                                                                                                 low = config.low_q,
                                                                                                 high = config.high_q)

            score_model.to('cpu')
            energy_model.to('cpu')
            pre_score_model = score_model
            pre_energy_model = energy_model
            pre_score_models.append(pre_score_model)
            pre_energy_models.append(pre_energy_model)

        if phase == "metric":
            freeze_params(metric_model.metric_net)
            
        if phase == "embed":
            freeze_params(embed_model.embed_net)

        if phase == "flow":
            pass

        model.to("cpu")

    return pre_score_models, pre_energy_models, score_model, energy_model, metric_model, embed_model, flow_model




def run_full_model_for_sweeping(
    config=None,
    run=None,            
    adata=None,
    values=None,
    conditions=None,
    dataset=None,
):
    cfg = getattr(run, "config", None) or config
    original_config = dict(cfg) if isinstance(cfg, dict) else None

    _, score_dataset, y = extract_score_dataset(adata, values, n_neighbors=cfg.n_neighbors, resolution=cfg.resolution)

    pre_score_model, pre_energy_model = None, None
    score_model = energy_model = metric_model = embed_model = flow_model = None
    pre_score_models, pre_energy_models = [], []
    metric_model_low_quantile = None
    metric_model_high_quantile = None
    metric_model_train_dataloader = None

    phase_list = ['score', 'energy'] * cfg.pita_steps + ['metric', 'embed', 'flow']

    weights = [1.0] * len(score_dataset)

    wandb_logger = WandbLogger(experiment=run)

    for i, phase in enumerate(phase_list):
        print(f"Running phase {phase}:.......")

        # build models
        if phase == 'score':
            score_model = build_score_pita(cfg, conditions, pre_score_model)
            # exponential schedule, descending across pita steps
            j = (i // 2)

            if j == config.pita_steps-1:
                #Final renormalization step, learn score over everything
                y *= 0
            
            betas = torch.exp(
                torch.linspace(torch.log(torch.tensor(cfg.score_beta_min, dtype=torch.float32)),
                               torch.log(torch.tensor(cfg.score_beta_max, dtype=torch.float32)),
                               cfg.pita_steps)
            ).tolist()
            betas.reverse()
            score_model.score_beta = betas[j]

        elif phase == 'energy':
            energy_model = build_energy_pita(cfg, conditions, score_model)

        elif phase == 'metric':
            metric_model = build_metric(cfg, conditions, energy_model)
            metric_model.low_quantile = metric_model_low_quantile
            metric_model.high_quantile = metric_model_high_quantile
            if getattr(cfg, "mfm_benchmark", False):
                metric_model.train_dataloader = metric_model_train_dataloader

        elif phase == 'embed':
            embed_model = build_embed(cfg, adata, conditions, metric_model)

        elif phase == 'flow':
            flow_model = build_flow(cfg, adata, conditions, embed_model)


        # dataloaders
        if phase in ['score', 'energy', 'metric']:
            sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
            train_dataset = TensorDataset(score_dataset, y)
            train_dataloader = DataLoader(
                train_dataset,
                batch_size=cfg.score_batch_size,
                sampler=sampler
            )
        else:
            if getattr(cfg, "fast_ot", False):
                update_epoch_rate = 50 if phase == "embed" else 10000
                train_dataset = ShufflingOTDataset(dataset, cfg.flow_batch_size, conditions, update_epoch_rate)
            else:
                train_dataset = ShufflingDataset(dataset, cfg.flow_batch_size, conditions)
            train_dataloader = DataLoader(
                train_dataset,
                batch_size=cfg.loader_batch_size,
                shuffle=True
            )

        # training
        trainer = build_trainer(cfg, wandb_logger, phase)
        model = {
            'score':  score_model,
            'energy': energy_model,
            'metric': metric_model,
            'embed':  embed_model,
            'flow':   flow_model
        }[phase]

        wandb_logger.watch(model, log="all")
        trainer.fit(model=model, train_dataloaders=train_dataloader)

        try:
            import wandb as _wandb
            _wandb.unwatch(model)
        except Exception:
            pass


        del trainer, train_dataloader
        try:
            del sampler
        except NameError:
            pass

        gc.collect()
        torch.cuda.empty_cache()
        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass

        model.eval()

        if phase == "score":
            freeze_params(score_model.score_net)

        elif phase == "energy":
            freeze_params(energy_model.energy_net)
            energy_model.score_model = None  
            train_dataset = TensorDataset(score_dataset, y)
            train_dataloader = DataLoader(train_dataset, batch_size=cfg.score_batch_size, shuffle=False)

            if getattr(cfg, "mfm_benchmark", False):
                metric_model_train_dataloader = train_dataloader

            sigma_scalar = np.median(energy_model.energy_sigma)
            weights = get_new_weights(energy_model,
                                      train_dataloader,
                                      sigma_scalar=sigma_scalar,
                                      low=cfg.pre_low_q,
                                      high=cfg.pre_high_q,
                                      weight_beta=cfg.weight_beta)

            metric_model_low_quantile, metric_model_high_quantile = get_energy_statistics_global(
                energy_model, train_dataloader,
                low=cfg.low_q, high=cfg.high_q)
            
            score_model.to('cpu')
            energy_model.to('cpu')
            pre_score_model = score_model
            pre_energy_model = energy_model
            pre_score_models.append(pre_score_model)
            pre_energy_models.append(pre_energy_model)

        elif phase == "metric":
            freeze_params(metric_model.metric_net)

        elif phase == "embed":
            freeze_params(embed_model.embed_net)

        elif phase == "flow":
            pass

        model.to('cpu')

    return pre_score_models, pre_energy_models, score_model, energy_model, metric_model, embed_model, flow_model