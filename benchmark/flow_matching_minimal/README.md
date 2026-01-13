Minimal, self-contained version of the flow-matching experiment originally in `notebooks/flow_matching`.

What’s included
- `train.py`: streamlined training loop (autoencoder → discriminator → geodesic flow matching).
- `autoencoder.py`, `discriminator.py`, `geodesic_fm.py`, `off_manifold.py`, `sampling.py`: the small modules the script depends on.
- `train.sh`: runnable shell script with the hyperparameters from the original notebook version.
- `requirements.txt`: minimal Python dependencies.

Usage
1) Install deps (in a clean venv/conda env):
```
pip install -r requirements.txt
```

2) Run training (expects the data npz used previously):
```
bash train.sh
```

Defaults assume the CITE data file lives at `../../../cite_all_D-100_d-3_pca.npz` relative to this folder (the repository root). Override by exporting `DATA_PATH=/path/to/cite_all_D-100_d-3_pca.npz` before running `train.sh`. Checkpoints and plots land under `checkpoints_minimal/` and `plots_minimal/`.

Weights & Biases logging
- Disabled by default. To log metrics offline (avoids network), run `WANDB_ARGS="--wandb --wandb_mode offline --wandb_project gaga-flow-matching" bash train.sh`.
- For online logging, switch `--wandb_mode online` and optionally set `WANDB_API_KEY`, `--wandb_entity`, or `--wandb_run_name`.

Reproducibility tips
- `train.sh` pins `--seed` and `--deterministic` so the GAGA baseline split and training are repeatable. The script saves `run_args.json` and `data_splits.npz` in `checkpoints_minimal/` for auditability.
- If you tweak hyperparameters, keep the seed fixed to match the original notebook’s trajectory.

Weights & Biases
- Logging is enabled in `train.sh` via `--wandb` (project defaults to `gaga-flow-matching`). Set `WANDB_API_KEY` and optionally `WANDB_ENTITY` before running.
- Images and the Wasserstein-1 metric are pushed to the same W&B run used for Lightning metrics. Override names/tags with the CLI flags in `train.sh`.
