import numpy as np
import torch
import wandb
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
    
def build_classifier(config):

    classifier_net = SimpleDenseNet(input_dim=config.pc_dim,
                                     output_dim=config.num_classes + 1,
                                     hidden_dims=[config.hidden_dim]*config.num_layers,
                                     layer_norm=True,)

    classifier_model = ClassifierNetTrainBase(classifier_net=classifier_net, config=config)
    return classifier_model

def build_metric(config, adata, classifier_model):

    if config.metric == "cfm":
        metric_model = MetricNetCFM(config=config)

    elif config.metric == "mfm":
        assert config.metric_max_epochs>5, "Need to train the metric tensor when you use MFM"
        #TODO: pass the dataloader?
        metric_model = MetricNetMFM(K = config.K,
                                    kappa=config.kappa,
                                    config=config)
    elif config.metric == "gaga":
        pass
    
    else:
        raise NotImplementedError(f"Metric model {config.metric} not implemented.")
    
    if config.use_finsler:
        metric_model = FinslerMetricNet(riemannian_metric_model=metric_model,
                                                classifier_model=classifier_model,
                                                config=config,
                                                tree=adata.uns['tree'],
                                                temp=config.finsler_temp)

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
                      output_dim=config.pc_dim,
                      num_freq=config.num_freq,
                      layer_norm=True,
                      hidden_dims=[config.hidden_dim]*config.num_layers,
                      rescale=config.rescale)

    timepoints = sorted(adata.obs['timepoint'].unique().tolist())
    t_global_min, t_global_max = min(timepoints), max(timepoints)
    
    if config.use_finsler:
        embed_model = FinslerEmbedNetTrainBase(metric_model=metric_model,
                                               geo_net=geo_net,
                                               embed_net=embed_net,
                                               config=config,
                                               t_global_min=t_global_min,
                                               t_global_max=t_global_max,
                                               sample_rescale=sample_rescale)
    else:
        embed_model = EmbedNetTrainBase(metric_model=metric_model,
                                    geo_net=geo_net,
                                    embed_net=embed_net,
                                    config=config,
                                    t_global_min=t_global_min,
                                    t_global_max=t_global_max,
                                    sample_rescale=sample_rescale)

    return embed_model

def build_flow(config, adata, embed_model):

    flow_net = SinNet(input_dim=config.pc_dim,
                      output_dim=config.pc_dim,
                      num_freq=config.num_freq,
                      layer_norm=True,
                      hidden_dims=[config.hidden_dim]*config.num_layers)

    timepoints = sorted(adata.obs['timepoint'].unique().tolist())
    t_global_min, t_global_max = min(timepoints), max(timepoints)
    sample_rescale = torch.from_numpy(adata.uns['std'])
    
    flow_model = FlowNetTrainBase(flow_net=flow_net,
                             embed_model=embed_model,
                             config=config,
                             t_global_min=t_global_min,
                             t_global_max=t_global_max,
                             sample_rescale=sample_rescale)

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



def run_full_model(config, project = None, adata = None, dataset = None):
    
    original_config = config.copy()
    X, y = extract_singleton_dataset(adata)

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
            metric_model = build_metric(config, adata, classifier_model)

        if phase == 'embed':
            embed_model = build_embed(config, adata, metric_model)

        if phase == 'flow':
            flow_model = build_flow(config, adata, embed_model)


        ### build dataset ###
        if phase in ['classifier', 'metric']:

            train_dataset = TensorDataset(X, y)
            train_dataloader = DataLoader(train_dataset, batch_size = config.score_batch_size, drop_last=True, shuffle=True)

            if phase == "metric" and config.metric == "mfm":
                #TODO: hard-coded?
                metric_model.train_dataloader = train_dataloader
            
        else:

            if config.fast_ot:
                #TODO: hard-coded?
                update_epoch_rate = 50 if phase == "embed" else 10000
                train_dataset = ShufflingOTDataset(dataset, config.flow_batch_size, update_epoch_rate)
            else:
                train_dataset = ShufflingDataset(dataset, config.flow_batch_size)
            train_dataloader = DataLoader(train_dataset, batch_size = config.loader_batch_size, shuffle=True)

        
        
        ### train ###
        trainer = build_trainer(config, wandb_logger, phase)
        model = {'classifier': classifier_model,
                 'metric': metric_model,
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
        
        ### cleanup ###
        model.eval()
        freeze_params(model)
        # model.to("cpu")

    return classifier_model, metric_model, embed_model, flow_model

