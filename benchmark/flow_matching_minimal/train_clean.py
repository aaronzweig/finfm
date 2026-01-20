import argparse
import json
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import ot
import torch
import pytorch_lightning as pl
try:
    from pytorch_lightning.loggers import WandbLogger
except Exception:
    WandbLogger = None
from torch.utils.data import DataLoader, Dataset

# EXTREME LAZY FIX FOR PATH MANAGEMENT
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from utils.hydra import *
from models.metric_models import MetricNetGAGA
from scripts.run_model import *
from utils.frozen import *
from omegaconf import OmegaConf


from autoencoder import (
    Autoencoder,
    PointCloudDataset,
    make_custom_collate_fn,
)
from discriminator import Discriminator
from geodesic_fm import GeodesicFlowMatching
from sampling import neg_sample_additive, sampling_rejection

def init_wandb_logger(args):
    if not args.wandb or args.wandb_mode == "disabled":
        return None
    if WandbLogger is None:
        raise ImportError("wandb is required for logging. Install it or disable --wandb.")
    run_name = args.wandb_run_name or f"flow-matching-minimal-seed{args.seed}"
    logger = WandbLogger(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=run_name,
        tags=args.wandb_tags,
        save_dir=args.wandb_dir,
        mode=args.wandb_mode,
    )
    logger.experiment.config.update(vars(args), allow_val_change=True)
    return logger



class PairDataset(Dataset):
    def __init__(self, x0, x1):
        self.x0 = torch.tensor(x0, dtype=torch.float32)
        self.x1 = torch.tensor(x1, dtype=torch.float32)

    def __len__(self):
        return max(len(self.x0), len(self.x1))

    def __getitem__(self, idx):
        return self.x0[idx % len(self.x0)], self.x1[idx % len(self.x1)]


def pair_collate_fn(batch):
    x0_batch = torch.stack([item[0] for item in batch])
    x1_batch = torch.stack([item[1] for item in batch])
    perm_x0 = torch.randperm(len(x0_batch))
    perm_x1 = torch.randperm(len(x1_batch))
    return x0_batch[perm_x0], x1_batch[perm_x1]

def encode_data(x, encoder, device, batch_size=256):
    encodings = []
    encoder.eval()
    encoder.to(device)
    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            batch_slice = x[i : i + batch_size]
            if isinstance(batch_slice, torch.Tensor):
                batch = batch_slice.to(device=device, dtype=torch.float32)
            else:
                batch = torch.as_tensor(batch_slice, dtype=torch.float32, device=device)
            encodings.append(encoder(batch, normalize=True).cpu().numpy())
    return np.concatenate(encodings, axis=0)


def train_autoencoder(pointcloud, distances, labels, args, device, logger=None):
    dataset = PointCloudDataset(pointcloud, distances)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, collate_fn=make_custom_collate_fn(dataset)
    )

    mean = pointcloud.mean(axis=0)
    std = pointcloud.std(axis=0)
    dist_std = np.std(distances.flatten())

    model = Autoencoder(
        data_dim=pointcloud.shape[1],
        latent_dim=args.ae_latent_dim,
        encoder_layer_width=args.ae_encoder_layer_width,
        dropout=args.ae_dropout,
        use_spectral_norm=args.ae_use_spectral_norm,
        mean=mean,
        std=std,
        dist_std=dist_std,
        dist_mse_decay=args.ae_dist_mse_decay,
        lr=args.ae_lr,
        weight_decay=args.ae_weight_decay,
        weights_dist=args.ae_weights_dist,
        weights_reconstr=args.ae_weights_reconstr,
        weights_cycle=args.ae_weights_cycle,
        weights_cycle_dist=args.ae_weights_cycle_dist,
    )

    trainer = pl.Trainer(
        max_epochs=args.ae_max_epochs,
        log_every_n_steps=args.ae_log_every_n_steps,
        accelerator=device,
        devices=1,
        logger=logger,
        deterministic=args.deterministic,
    )
    trainer.fit(model, train_dataloaders=loader)
    return model

def train_discriminator(
    x_pos,
    x_neg,
    encoder,
    batch_size=128,
    max_epochs=100,
    lr=1e-3,
    weight_decay=1e-4,
    layer_widths=None,
    logger=None,
    deterministic=False,
    checkpoint_dir=None,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.backends.mps.is_available():
        device = "mps"

    X = torch.cat([x_pos, x_neg], dim=0)
    Y = torch.cat(
        [torch.ones(x_pos.shape[0], dtype=torch.long), torch.zeros(x_neg.shape[0], dtype=torch.long)],
        dim=0,
    )

    dataset = torch.utils.data.TensorDataset(X, Y)

    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = Discriminator(
        in_dim=x_pos.shape[1],
        lr=lr,
        weight_decay=weight_decay,
        layer_widths=layer_widths,
    )
    if checkpoint_dir is not None:
        os.makedirs(checkpoint_dir, exist_ok=True)
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        log_every_n_steps=10,
        accelerator=device,
        devices=1,
        logger=logger,
        deterministic=deterministic,
    )
    trainer.fit(model, train_dataloaders=loader)
    return model

from eval.eval import *
import scanpy as sc
def main(args):
    wandb_logger = init_wandb_logger(args)

    device = "cuda"

    os.makedirs(args.plots_save_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    #TODO: change to load two adatas: one with all data, one with a heldout timepoint and phate distances
    #saved in adata.obsp

    adata = sc.read_h5ad(os.path.split(args.data_path)[0] + "/cite_100.h5ad")
    x = adata.obsm['X_pca']
    x_dist = adata.obsp['dist']
    # x_dist = np.load(args.data_path)['dist']
    labels = np.array(adata.obs['day_class'].tolist())

    leave_out_idx = np.where(labels != args.test_group)[0]
    ae_model = train_autoencoder(
        x[leave_out_idx],
        x_dist[leave_out_idx][:, leave_out_idx],
        labels[leave_out_idx],
        args,
        device,
        logger=wandb_logger,
    )

    x_encodings = encode_data(x, ae_model, device)

    train_encodings_leave_out = x_encodings[labels != args.test_group]
    x_noisy = neg_sample_additive(train_encodings_leave_out, args.noise_levels, seed=args.seed)
    if args.sampling_rejection:
        rejected_mask = sampling_rejection(
            train_encodings_leave_out,
            x_noisy,
            method=args.sampling_rejection_method,
            k=args.sampling_rejection_k,
            threshold=args.sampling_rejection_threshold,
        )
        x_noisy = x_noisy[~rejected_mask]

    wd_model = train_discriminator(
        torch.tensor(train_encodings_leave_out, dtype=torch.float32),
        torch.tensor(x_noisy, dtype=torch.float32),
        ae_model,
        batch_size=args.disc_batch_size,
        max_epochs=args.disc_max_epochs,
        lr=args.disc_lr,
        weight_decay=args.disc_weight_decay,
        layer_widths=args.disc_layer_widths,
        logger=wandb_logger,
        deterministic=args.deterministic,
        checkpoint_dir=args.checkpoint_dir,
    )
    ae_model.to(device)
    wd_model.to(device)

    start_pts = x[labels == args.start_group]
    end_pts = x[labels == args.end_group]

    def align_pairs(a, b, seed):
        """Match start/end sets to equal length with a deterministic permutation."""
        rng = np.random.default_rng(seed)
        perm_a = rng.permutation(len(a))
        perm_b = rng.permutation(len(b))
        m = min(len(a), len(b))
        return a[perm_a[:m]], b[perm_b[:m]]

    start_pts, end_pts = align_pairs(start_pts, end_pts, args.seed)

    dataset = PairDataset(start_pts, end_pts)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=pair_collate_fn)

    ae_model.eval()
    wd_model.eval()
    enc_func = lambda x: ae_model(x, normalize=True)
    def make_offmanifold(encoder, discriminator, disc_factor=5.0):

        def _fn(x):
            z = encoder(x)
            penalty = torch.exp(disc_factor * (1 - discriminator.positive_prob(z)))
            return torch.cat([z, penalty.unsqueeze(1)], dim=1)

        return _fn
    ofm = make_offmanifold(enc_func, wd_model, disc_factor=args.disc_factor)

    ###########
    config = load_config()
    OmegaConf.set_struct(config, False)
    config.pc_dim = 100
    config.embed_max_epochs = 2 #DEBUG
    adata_train = adata[adata.obs['day_class'].isin([args.start_group, args.end_group])]
    adata_train.obs['timepoint'] = adata_train.obs['day_class']
    timepoints = adata_train.obs['timepoint'].unique()
    train_dataloader = build_paired_dataloader(config, adata_train)
    freeze_params(ae_model)
    freeze_params(wd_model)
    # metric_model = MetricNetGAGA(config=config,
    #                              encoder=ae_model,
    #                              discriminator=wd_model)
    metric_model = MetricNetCFM(config=config)
    embed_model = build_embed(config, timepoints, metric_model)
    trainer = build_trainer(config, None, "embed")
    trainer.fit(model=embed_model, train_dataloaders=train_dataloader)

    samples = []
    total = 0
    t = args.test_group
    while total < 20000:
        batch = next(iter(train_dataloader))
        x = embed_model.sample_geodesic_time(batch, t)
        samples.append(x)
        total += x.shape[0]
    generated_data = torch.cat(samples, dim=0).detach().cpu().numpy()
    real_data = adata[adata.obs['day_class'] == args.test_group].obsm['X_pca']
    ###########





    # gbmodel = GeodesicFlowMatching(
    #     func=ofm,
    #     encoder=enc_func,
    #     input_dim=x.shape[1],
    #     hidden_dim=args.hidden_dim,
    #     scale_factor=args.scale_factor,
    #     embed_t=args.embed_t,
    #     num_layers=args.num_layers,
    #     n_tsteps=args.n_tsteps,
    #     lr=args.lr,
    #     weight_decay=args.weight_decay,
    #     flow_weight=args.flow_weight,
    #     length_weight=args.length_weight,
    # )

    # trainer = pl.Trainer(
    #     max_epochs=args.max_epochs,
    #     log_every_n_steps=args.log_every_n_steps,
    #     accelerator=device,
    #     devices=1,
    #     logger=wandb_logger,
    #     deterministic=args.deterministic,
    # )
    # trainer.fit(gbmodel, train_dataloaders=loader)
    # gbmodel.to(device)
    # gbmodel.eval()

    # print("finished training")

    # ts_eval = torch.linspace(0, 1, args.n_tsteps, device=device)
    # gb_trajs = gbmodel.cc(
    #     torch.tensor(start_pts, dtype=torch.float32, device=device),
    #     torch.tensor(end_pts, dtype=torch.float32, device=device),
    #     ts_eval,
    # )

    # print(x.shape, start_pts.shape, end_pts.shape, ts_eval.shape)

    # print("sampled traj")

    # real_idx = np.where(labels == args.test_group)[0]
    # real_data = x[real_idx]
    # generated_data = gb_trajs.detach().cpu().reshape(-1, gb_trajs.shape[-1]).numpy()





    print("now calc wass")
    print(real_data.shape)
    print(generated_data.shape)

    subset = np.random.choice(generated_data.shape[0], 20000, replace=False)

    print(generated_data[subset].shape)

    w1 = ot_dist(torch.from_numpy(generated_data[subset]), torch.from_numpy(real_data), library="geomloss")

    # w1 = eval_wasserstein(generated_data[subset], real_data)

    print(w1)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Minimal flow-matching training script")
    parser.add_argument("--data_path", type=str, default="../../../cite_all_D-100_d-3_pca.npz")
    parser.add_argument("--plots_save_dir", type=str, default="./plots_minimal")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints_minimal")
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--test_size", type=float, default=0.1)
    parser.add_argument("--val_size", type=float, default=0.1)
    parser.add_argument("--neg_method", type=str, default="add")
    parser.add_argument("--noise_levels", type=float, nargs="+", default=[0.5, 1.0])
    parser.add_argument("--sampling_rejection", action="store_true")
    parser.add_argument("--sampling_rejection_method", type=str, default="sugar")
    parser.add_argument("--sampling_rejection_k", type=int, default=20)
    parser.add_argument("--sampling_rejection_threshold", type=float, default=0.2)
    parser.add_argument("--disc_batch_size", type=int, default=128)
    parser.add_argument("--disc_layer_widths", type=int, nargs="+", default=[256, 128, 64])
    parser.add_argument("--disc_lr", type=float, default=1e-3)
    parser.add_argument("--disc_weight_decay", type=float, default=1e-4)
    parser.add_argument("--disc_max_epochs", type=int, default=100)
    parser.add_argument("--disc_factor", type=float, default=10.0)
    parser.add_argument("--alpha", type=float, default=8.0)
    parser.add_argument("--start_group", type=int, default=0)
    parser.add_argument("--end_group", type=int, default=2)
    parser.add_argument("--test_group", type=int, default=1)
    parser.add_argument("--range_size", type=float, default=0.3)
    parser.add_argument("--num_samples", type=int, default=128)
    parser.add_argument("--train_autoencoder", action="store_true")
    parser.add_argument("--ae_latent_dim", type=int, default=3)
    parser.add_argument("--ae_batch_norm", action="store_true")
    parser.add_argument("--ae_use_spectral_norm", action="store_true")
    parser.add_argument("--ae_dropout", type=float, default=0.2)
    parser.add_argument("--ae_dist_mse_decay", type=float, default=0.0)
    parser.add_argument("--ae_weights_dist", type=float, default=77.4)
    parser.add_argument("--ae_weights_reconstr", type=float, default=0.32)
    parser.add_argument("--ae_weights_cycle", type=float, default=1.0)
    parser.add_argument("--ae_weights_cycle_dist", type=float, default=0.0)
    parser.add_argument("--ae_lr", type=float, default=1e-3)
    parser.add_argument("--ae_weight_decay", type=float, default=1e-4)
    parser.add_argument("--ae_max_epochs", type=int, default=100)
    parser.add_argument("--ae_log_every_n_steps", type=int, default=50)
    parser.add_argument("--ae_early_stop_patience", type=int, default=50)
    parser.add_argument("--ae_encoder_layer_width", type=int, nargs="+", default=[256, 128, 64])
    parser.add_argument("--ae_decoder_layer_width", type=int, nargs="+", default=[64, 128, 256])
    parser.add_argument("--ae_activation", type=str, default="relu")
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--scale_factor", type=float, default=1.0)
    parser.add_argument("--embed_t", action="store_true")
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--n_tsteps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--flow_weight", type=float, default=1.0)
    parser.add_argument("--length_weight", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=150)
    parser.add_argument("--max_epochs", type=int, default=300)
    parser.add_argument("--log_every_n_steps", type=int, default=20)
    parser.add_argument("--deterministic", action="store_true", help="Enforce deterministic training for reproducibility")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="gaga-flow-matching", help="W&B project name")
    parser.add_argument("--wandb_entity", type=str, default=None, help="Optional W&B entity")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="Optional W&B run name override")
    parser.add_argument("--wandb_tags", nargs="*", default=[], help="Optional list of W&B tags")
    parser.add_argument("--wandb_dir", type=str, default="./wandb", help="Optional W&B save directory")
    parser.add_argument(
        "--wandb_mode",
        type=str,
        default="disabled",
        choices=["online", "offline", "disabled"],
        help="W&B mode (disabled by default to avoid network requirements).",
    )

    args = parser.parse_args()
    main(args)
