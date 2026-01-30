import numpy as np
import torch
import wandb
import os
from torch.utils.data import DataLoader
import pytorch_lightning as pl

import os 
import sys
from pytorch_lightning.loggers import WandbLogger

from scripts.run_model import *
from eval.eval import *
import time
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb

@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # 1. Setup WandB (Crucial for sweeps to group runs)
    wandb.init(
        project=cfg.project,
        entity="az831-new-york-genome-center",
        config=OmegaConf.to_container(cfg, resolve=True),
        reinit=True
    )
    
    _, _, _, _, w1_scores = train(cfg, cfg.project)
    
    # 4. Log final metric for the sweep to target
    wandb.log({"w1_avg_error": torch.mean(w1_scores)}) 

if __name__ == "__main__":
    main()