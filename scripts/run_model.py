import numpy as np
import torch
import wandb
import sys
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from torch.utils.data import Dataset, Sampler, DataLoader
import pytorch_lightning as pl
import torch
from torch.utils.data import TensorDataset, Sampler, DataLoader
import pytorch_lightning as pl
from models.fm_models import *
from models.metric_models import *
from models.embed_models import *
from utils.preprocess import *
from datasets.dataset import *
from utils.frozen import *
from utils.callback import *
from models.modules import *
from models.ema import *
from models.classifier_models import *
from datasets.process import *
from omegaconf import OmegaConf
from eval.eval import *
    
def build_classifier(config):
    classifier_net = SimpleDenseNet(input_dim=config.pc_dim,
                                     output_dim=config.num_classes+1,
                                     hidden_dims=[config.classifier.hidden_dim]*config.classifier.num_layers,
                                     layer_norm=True)

    classifier_model = ClassifierNetTrainBase(classifier_net=classifier_net, config=config)
    return classifier_model

def build_metric(config, tree, classifier_model):

    if config.metric == "cfm":
        if not config.finsler.use:
            metric_model = MetricNetCFM(config=config)
        else:
            metric_model = FinslerCFM(config=config,
                                      classifier_model=classifier_model,
                                      tree=torch.from_numpy(tree).float(),
                                      temp=config.finsler.temp,
                                      lamb=config.finsler.lamb,
                                      )

    elif config.metric == "mfm":
        #TODO: pass the dataloader?
        if not config.finsler.use:
            metric_model = MetricNetMFM(K = config.mfm.K,
                                        kappa=config.mfm.kappa,
                                        alpha=config.mfm.alpha,
                                        epsilon=config.mfm.epsilon,
                                        config=config)
        else:
            metric_model = FinslerMFM(config=config,
                                      classifier_model=classifier_model,
                                      tree=torch.from_numpy(tree).float(),
                                      temp=config.finsler.temp,
                                      lamb=config.finsler.lamb,
                                      K = config.mfm.K,
                                      kappa=config.mfm.kappa,
                                      alpha=config.mfm.alpha,
                                      epsilon=config.mfm.epsilon,
                                      )
    elif config.metric == "gaga":
        pass
    
    else:
        raise NotImplementedError(f"Metric model {config.metric} not implemented.")

    return metric_model

def build_embed(config, timepoints, metric_model):

    embed_net = SimpleEmbedNet(input_dim=config.pc_dim,
                  output_dim=config.latent_dim,
                  layer_norm=True,
                  hidden_dims=[config.hidden_dim]*config.num_layers,
                  rescale=config.rescale,
                  skip=config.skip)

    geo_net = SinNet(input_dim=2 * config.pc_dim,
                      output_dim=config.pc_dim,
                      num_freq=config.num_freq,
                      layer_norm=True,
                      hidden_dims=[config.hidden_dim]*config.num_layers,
                      rescale=config.rescale)

    t_global_min, t_global_max = min(timepoints), max(timepoints)
    
    if config.finsler.use:
        embed_model = FinslerEmbedNetTrainBase(metric_model=metric_model,
                                               geo_net=geo_net,
                                               embed_net=embed_net,
                                               config=config,
                                               t_global_min=t_global_min,
                                               t_global_max=t_global_max)
    else:
        embed_model = EmbedNetTrainBase(metric_model=metric_model,
                                    geo_net=geo_net,
                                    embed_net=embed_net,
                                    config=config,
                                    t_global_min=t_global_min,
                                    t_global_max=t_global_max)

    return embed_model

def build_flow(config, timepoints, embed_model):

    flow_net = SinNet(input_dim=config.pc_dim,
                      output_dim=config.pc_dim,
                      num_freq=config.num_freq,
                      layer_norm=True,
                      hidden_dims=[config.hidden_dim]*config.num_layers)

    t_global_min, t_global_max = min(timepoints), max(timepoints)
    
    flow_model = FlowNetTrainBase(flow_net=flow_net,
                             embed_model=embed_model,
                             config=config,
                             t_global_min=t_global_min,
                             t_global_max=t_global_max)

    return flow_model

def build_trainer(config, wandb_logger, phase):
    callbacks = []
    if phase == "classifier":
        max_epochs = config.classifier_max_epochs
    elif phase == "metric":
        max_epochs = config.metric_max_epochs
    elif phase == "embed":
        max_epochs = config.embed_max_epochs
        callbacks.append(DatasetUpdateCallback())
    elif phase == "flow":
        max_epochs = config.flow_max_epochs
    else:
        raise NotImplementedError(f"Phase {phase} not implemented.")
    trainer = pl.Trainer(
        accelerator="cpu" if config.force_cpu else "gpu", 
        logger=wandb_logger,
        callbacks = callbacks,
        log_every_n_steps=1, 
        max_epochs=max_epochs, 
        gradient_clip_val=config.gradient_clip_val,
        enable_checkpointing=False,
        detect_anomaly=config.detect_anomaly,
        enable_progress_bar=False,
        )
    return trainer

def build_singleton_dataloader(config, adata):
    X, y = extract_singleton_dataset(adata)
    train_dataset = TensorDataset(X, y)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size = config.score_batch_size,
                                  drop_last=True, 
                                  shuffle=True)

    return train_dataloader

def build_paired_dataloader(config, adata):
    dataset = extract_paired_dataset(adata)

    if config.fast_ot:
        #TODO: hard-coded?  Actually validate if this works sensibly
        #TODO: we don't need to update during flow btw
        update_epoch_rate = 50
        train_dataset = ShufflingOTDataset(dataset, config.flow_batch_size, update_epoch_rate)
    else:
        train_dataset = ShufflingDataset(dataset, config.flow_batch_size)
    train_dataloader = DataLoader(train_dataset, batch_size = config.loader_batch_size, shuffle=True)
    return train_dataloader

def run_full_model(config, project, singleton_dataloader, paired_dataloader, timepoints, tree, wandb_logger=None):
    """Train all four phases (classifier, metric, embed, flow).

    Args:
        wandb_logger: Optional external WandbLogger (e.g. from sweep.py).
            If provided, this single logger is reused for every phase — no
            per-phase wandb.init() calls are made.  When None, behaviour
            falls back to config.use_wandb (notebook path with per-phase runs).
    """
    classifier_model = build_classifier(config)
    metric_model = build_metric(config, tree, classifier_model)
    embed_model = build_embed(config, timepoints, metric_model)
    flow_model = build_flow(config, timepoints, embed_model)

    if config.normalize:
        X = singleton_dataloader.dataset.tensors[0]
        mean = torch.mean(X, dim=0, keepdim = True)
        std = torch.max(torch.std(X, dim=0))
        for model in [classifier_model, metric_model, embed_model, flow_model]:
            model.mean = mean
            model.std = std

    if config.metric == "mfm":
        #TODO: hard-coded?
        metric_model.train_dataloader = singleton_dataloader

    if config.no_learning:
        assert config.metric == "cfm" and not config.finsler.use, "you need to learn a metric"
        return classifier_model, metric_model, embed_model, flow_model

    phase_list =  ['classifier', 'metric', 'embed', 'flow']

    for i, phase in enumerate(phase_list):

        print(f"Running phase {phase}:.......")

        # Determine the logger for this phase
        if wandb_logger is not None:
            # External logger (sweep path) — reuse the single run
            phase_logger = wandb_logger
        elif config.use_wandb:
            # Notebook path — separate wandb run per phase
            config_dict = OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
            phase_logger = WandbLogger(project=project, name=phase, log_model=True)
            wandb.init(config=config_dict, project=project, reinit=True)
        else:
            phase_logger = False

        ### train ###
        trainer = build_trainer(config, phase_logger, phase)
        model = {'classifier': classifier_model,
                 'metric': metric_model,
                 'embed': embed_model,
                 'flow': flow_model}[phase]

        train_dataloader = singleton_dataloader if phase in ["classifier", "metric"] else paired_dataloader

        trainer.fit(model=model, train_dataloaders=train_dataloader)

        # Cleanup per-phase wandb run (notebook path only)
        if wandb_logger is None and config.use_wandb:
            wandb.finish()

        ### cleanup ###
        model.eval()
        freeze_params(model)

    return classifier_model, metric_model, embed_model, flow_model

def remove_all_forward_hooks(model):
    for module in model.modules():
        module._forward_hooks.clear()

def train(config, project, wandb_logger=None):

    adata = process_data(pc_dim=config.pc_dim, data=config.dataset)
    timepoints = sorted(adata.obs['timepoint'].unique().tolist())
    tree = adata.uns['tree']

    t0, t1 = timepoints[config.t0_index], timepoints[config.t1_index]

    adata = adata[(adata.obs['timepoint'] >= t0) & (adata.obs['timepoint'] <= t1)]
    adata_train = adata[adata.obs['timepoint'].isin([t0, t1])]

    singleton_dataloader = build_singleton_dataloader(config, adata_train)
    paired_dataloader = build_paired_dataloader(config, adata_train)

    classifier_model, metric_model, embed_model, flow_model = run_full_model(config=config,
                                                                            project=project,
                                                                            singleton_dataloader=singleton_dataloader,
                                                                            paired_dataloader=paired_dataloader,
                                                                            timepoints=timepoints,
                                                                            tree=tree,
                                                                            wandb_logger=wandb_logger)

    remove_all_forward_hooks(classifier_model)
    remove_all_forward_hooks(metric_model)
    remove_all_forward_hooks(embed_model)
    remove_all_forward_hooks(flow_model)

    w1_scores = []
    for index in range(config.t0_index + 1, config.t1_index):
        t = timepoints[index]
        w1 = predict(embed_model, adata, t0, t, t1, num_traj=6000, library="pot")
        w1_scores.append(w1)
    w1_scores = torch.tensor(w1_scores)
    
    return classifier_model, metric_model, embed_model, flow_model, w1_scores