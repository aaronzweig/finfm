import argparse
import os
import yaml
import torch
import wandb
from functools import partial
from omegaconf import OmegaConf
from pytorch_lightning.loggers import WandbLogger

from scripts.run_model import train
from utils.hydra import load_config


def run_trial(dataset, overrides=None):
    """Single sweep trial — called by wandb.agent for each sampled config."""
    run = wandb.init(reinit=True)

    # Build base config from Hydra, apply fixed CLI overrides, then overlay wandb's sampled params
    cfg = load_config(overrides=[f"dataset={dataset}"])
    OmegaConf.set_struct(cfg, False)
    cfg.use_wandb = False
    for o in (overrides or []):
        key, value = o.split("=", 1)
        OmegaConf.update(cfg, key, yaml.safe_load(value))
    for key, value in dict(wandb.config).items():
        OmegaConf.update(cfg, key, value)

    # Log the full merged config so it's visible on the wandb dashboard
    wandb.config.update(OmegaConf.to_container(cfg, resolve=True), allow_val_change=True)

    wandb_logger = WandbLogger(experiment=run)
    _, _, _, _, w1_scores, w1_val_scores = train(cfg, cfg.project, wandb_logger=wandb_logger)

    avg = torch.mean(w1_scores).item()
    val_avg = torch.mean(w1_val_scores).item()
    wandb.log({"w1_avg": avg})
    wandb.summary["w1_avg"] = avg
    wandb.log({"w1_val_avg": val_avg})
    wandb.summary["w1_val_avg"] = val_avg

    wandb.finish()


def main():
    parser = argparse.ArgumentParser(description="Register a W&B sweep and run the agent.")
    parser.add_argument("--sweep_config", default="configs/sweeps/cfm_finsler.yaml",
                        help="Path to the W&B sweep YAML config")
    parser.add_argument("--dataset", default="zebrafish",
                        help="Dataset config to use (e.g. zebrafish, cite)")
    parser.add_argument("--count", type=int, default=None,
                        help="Max number of sweep trials (default: unlimited)")
    parser.add_argument("overrides", nargs="*",
                        help="Fixed config overrides applied every trial (e.g. t0_index=1 lr=5e-4)")
    args = parser.parse_args()

    # Load sweep config; drop command/program since wandb.agent calls run_trial directly
    with open(args.sweep_config) as f:
        sweep_config = yaml.safe_load(f)
    sweep_config.pop("command", None)
    sweep_config.pop("program", None)

    # Resolve project name from the dataset config
    base_cfg = load_config(overrides=[f"dataset={args.dataset}"])
    project = base_cfg.project

    # Validate that all override keys exist in config to catch typos early
    _MISSING = object()
    for o in (args.overrides or []):
        key, _ = o.split("=", 1)
        if OmegaConf.select(base_cfg, key, default=_MISSING) is _MISSING:
            parser.error(f"override key '{key}' not found in config")

    sweep_id = wandb.sweep(sweep_config, project=project)
    wandb.agent(sweep_id, function=partial(run_trial, args.dataset, args.overrides), count=args.count)


if __name__ == "__main__":
    main()
