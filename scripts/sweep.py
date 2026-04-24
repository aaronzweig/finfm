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


def get_best_run_config(sweep_id, entity, project, metric):
    """Fetch the best run's config from a completed W&B sweep."""
    api = wandb.Api()
    sweep = api.sweep(f"{entity}/{project}/{sweep_id}")
    goal = sweep.config.get("metric", {}).get("goal", "minimize")
    best_run = None
    best_val = None
    for run in sweep.runs:
        val = run.summary.get(metric)
        if val is None:
            continue
        if best_val is None or (goal == "minimize" and val < best_val) or (goal == "maximize" and val > best_val):
            best_val = val
            best_run = run
    if best_run is None:
        raise RuntimeError(f"No runs with metric '{metric}' found in sweep {sweep_id}")
    # Extract only the keys that were swept (i.e. defined in sweep.config["parameters"])
    swept_keys = set(sweep.config.get("parameters", {}).keys())
    best_params = {k: v for k, v in best_run.config.items() if k in swept_keys}
    return best_params, best_val, best_run.name, sweep.config


def main():
    parser = argparse.ArgumentParser(description="Register a W&B sweep and run the agent.")
    parser.add_argument("--sweep_config", default="configs/sweeps/cfm_finsler.yaml",
                        help="Path to the W&B sweep YAML config")
    parser.add_argument("--dataset", default="zebrafish",
                        help="Dataset config to use (e.g. zebrafish, cite)")
    parser.add_argument("--count", type=int, default=None,
                        help="Max number of sweep trials (default: unlimited)")
    # --from_sweep mode: pull best params from a completed sweep and re-run over multiple seeds
    parser.add_argument("--from_sweep", default=None, metavar="SWEEP_ID",
                        help="W&B sweep ID whose best run will be used as fixed params")
    parser.add_argument("--seeds", type=int, default=10,
                        help="Number of seeds to run when using --from_sweep (default: 10)")
    parser.add_argument("overrides", nargs="*",
                        help="Fixed config overrides applied every trial (e.g. t0_index=1 lr=5e-4)")
    args = parser.parse_args()

    # Resolve project name from the dataset config
    base_cfg = load_config(overrides=[f"dataset={args.dataset}"])
    project = base_cfg.project

    if args.from_sweep:
        # --- Mode: fetch best params from a completed sweep and grid over seeds ---
        api = wandb.Api()
        # Determine entity from the existing wandb settings or the sweep itself
        sweep_obj = api.sweep(f"{project}/{args.from_sweep}")
        entity = sweep_obj.entity
        metric_name = sweep_obj.config.get("metric", {}).get("name", "w1_val_avg")

        best_params, best_val, best_run_name, orig_sweep_cfg = get_best_run_config(
            args.from_sweep, entity, project, metric_name
        )

        print(f"\nBest run: {best_run_name}  ({metric_name} = {best_val:.4f})")
        print("Best hyperparameters:")
        for k, v in best_params.items():
            print(f"  {k}: {v}")
        print(f"\nLaunching grid sweep over seeds 1..{args.seeds} with these fixed params.\n")

        # Build in-memory grid sweep config: fix all best params, sweep only seed
        sweep_config = {
            "method": "grid",
            "metric": orig_sweep_cfg.get("metric", {"goal": "minimize", "name": metric_name}),
            "parameters": {
                **{k: {"value": v} for k, v in best_params.items()},
                "seed": {"values": list(range(1, args.seeds + 1))},
                "seeded": {"value": True},
            },
        }
        # Remove seeded=False if the original sweep set it; we override above
        sweep_config["parameters"].pop("seeded", None)
        sweep_config["parameters"]["seeded"] = {"value": True}
    else:
        # --- Default mode: run a sweep from a YAML config file ---
        with open(args.sweep_config) as f:
            sweep_config = yaml.safe_load(f)
        sweep_config.pop("command", None)
        sweep_config.pop("program", None)

    # Validate that all override keys exist in config to catch typos early
    _MISSING = object()
    for o in (args.overrides or []):
        key, _ = o.split("=", 1)
        if OmegaConf.select(base_cfg, key, default=_MISSING) is _MISSING:
            parser.error(f"override key '{key}' not found in config")

    sweep_id = wandb.sweep(sweep_config, project=project)
    # Print in a grep-friendly format so shell scripts can capture it
    print(f"SWEEP_ID={sweep_id}", flush=True)
    wandb.agent(sweep_id, function=partial(run_trial, args.dataset, args.overrides), count=args.count)


if __name__ == "__main__":
    main()
