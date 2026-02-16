import numpy as np
import torch
import wandb
import random
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from omegaconf import OmegaConf
from scipy.sparse import issparse

from models.gaga import (
    Autoencoder,
    Discriminator,
    PointCloudDataset,
    make_custom_collate_fn,
    neg_sample_additive,
    sampling_rejection,
    encode_data,
)
from models.metric_models import MetricNetGAGA
from models.embed_models import EmbedNetTrainBase
from models.modules import SimpleEmbedNet, SinNet
from datasets.process import process_data, extract_paired_dataset
from datasets.dataset import ShufflingDataset, ShufflingOTDataset
from utils.frozen import freeze_params
from utils.callback import BestModelCallback
from eval.eval import predict
from scripts.run_model import build_embed, build_paired_dataloader, build_trainer, remove_all_forward_hooks


# --- Phase 1: Autoencoder ---

def build_ae_dataloader(config, X_pca, distances):
    """Build dataloader for autoencoder training with distance matrix."""
    dataset = PointCloudDataset(X_pca, distances)
    loader = DataLoader(
        dataset,
        batch_size=config.gaga.ae_batch_size,
        shuffle=True,
        collate_fn=make_custom_collate_fn(dataset),
    )
    return loader


def build_autoencoder(config, dist_std):
    """Build autoencoder model."""
    ae_model = Autoencoder(
        config=config,
        latent_dim=config.gaga.ae_latent_dim,
        encoder_layer_width=config.gaga.ae_hidden_dims,
        dropout=config.gaga.ae_dropout,
        use_spectral_norm=config.gaga.ae_use_spectral_norm,
        dist_std=dist_std,
        dist_mse_decay=config.gaga.ae_dist_mse_decay,
        weights_dist=config.gaga.ae_weights_dist,
        ae_lr=config.gaga.ae_lr,
        ae_weight_decay=config.gaga.ae_weight_decay,
    )
    return ae_model


def train_autoencoder(config, ae_model, ae_dataloader, wandb_logger=None):
    """Train the autoencoder on point cloud data with distance preservation."""
    logger = wandb_logger if wandb_logger is not None else (
        WandbLogger(project="gaga", name="autoencoder") if config.use_wandb else False
    )
    trainer = pl.Trainer(
        max_epochs=config.gaga.ae_max_epochs,
        log_every_n_steps=1,
        accelerator="cpu" if config.force_cpu else "gpu",
        logger=logger,
        enable_checkpointing=False,
        enable_progress_bar=False,
        gradient_clip_val=config.gradient_clip_val,
    )
    trainer.fit(ae_model, train_dataloaders=ae_dataloader)
    return ae_model


# --- Phase 2: Discriminator ---

def build_and_train_discriminator(config, ae_model, X_pca, wandb_logger=None):
    """Encode data, generate negatives, and train discriminator."""
    device = "cpu" if config.force_cpu else "cuda"

    # Encode training data through frozen autoencoder
    encodings = encode_data(X_pca, ae_model, device)

    # Generate negative samples (off-manifold)
    noise_levels = list(config.gaga.noise_levels)
    negatives = neg_sample_additive(encodings, noise_levels, seed=config.seed)

    # Optional rejection sampling
    if config.gaga.sampling_rejection:
        rejected_mask = sampling_rejection(
            encodings,
            negatives,
            method=config.gaga.sampling_rejection_method,
            k=config.gaga.sampling_rejection_k,
            threshold=config.gaga.sampling_rejection_threshold,
        )
        negatives = negatives[~rejected_mask]

    # Build discriminator dataloader
    pos = torch.tensor(encodings, dtype=torch.float32)
    neg = torch.tensor(negatives, dtype=torch.float32)
    X = torch.cat([pos, neg], dim=0)
    Y = torch.cat([
        torch.ones(len(pos), dtype=torch.long),
        torch.zeros(len(neg), dtype=torch.long),
    ])
    disc_loader = DataLoader(
        TensorDataset(X, Y),
        batch_size=config.gaga.disc_batch_size,
        shuffle=True,
    )

    # Build and train discriminator
    disc_model = Discriminator(
        in_dim=config.gaga.ae_latent_dim,
        layer_widths=config.gaga.disc_hidden_dims,
        lr=config.gaga.disc_lr,
        weight_decay=config.gaga.disc_weight_decay,
        dropout=config.gaga.disc_dropout,
        use_spectral_norm=config.gaga.disc_use_spectral_norm,
    )

    logger = wandb_logger if wandb_logger is not None else (
        WandbLogger(project="gaga", name="discriminator") if config.use_wandb else False
    )
    trainer = pl.Trainer(
        max_epochs=config.gaga.disc_max_epochs,
        log_every_n_steps=1,
        accelerator="cpu" if config.force_cpu else "gpu",
        logger=logger,
        enable_checkpointing=False,
        enable_progress_bar=False,
    )
    trainer.fit(disc_model, train_dataloaders=disc_loader)
    return disc_model


# --- Main pipeline ---

def run_gaga_model(config, project, adata_train, timepoints, wandb_logger=None):
    """Run the full GAGA pipeline: autoencoder -> discriminator -> embed."""

    X_pca = adata_train.obsm['X_pca']

    # Phase 0: Compute normalization stats (ModelBase-style)
    X_tensor = torch.from_numpy(X_pca).float()
    mean = torch.mean(X_tensor, dim=0, keepdim=True)
    std = torch.max(torch.std(X_tensor, dim=0)) * np.sqrt(config.pc_dim)

    # Get distance matrix
    dist_key = config.gaga.distance_key
    distances = adata_train.obsp[dist_key]
    if issparse(distances):
        distances = distances.toarray()
    distances = np.asarray(distances, dtype=np.float32)
    dist_std = np.std(distances.flatten())

    # Phase 1: Autoencoder
    print("Phase 1: Training autoencoder...")
    ae_model = build_autoencoder(config, dist_std)
    # Set shared normalization before training
    ae_model.mean = mean
    ae_model.std = std

    ae_dataloader = build_ae_dataloader(config, X_pca, distances)
    train_autoencoder(config, ae_model, ae_dataloader, wandb_logger)
    ae_model.eval()
    freeze_params(ae_model)

    # Phase 2: Discriminator
    print("Phase 2: Training discriminator...")
    disc_model = build_and_train_discriminator(config, ae_model, X_pca, wandb_logger)
    disc_model.eval()
    freeze_params(disc_model)

    # Phase 3: Embed
    print("Phase 3: Training embed model...")
    metric_model = MetricNetGAGA(
        config=config,
        encoder=ae_model,
        discriminator=disc_model,
        disc_factor=config.gaga.disc_factor,
    )
    embed_model = build_embed(config, timepoints, metric_model)

    # Set shared normalization on metric and embed models
    for model in [metric_model, embed_model]:
        model.mean = mean
        model.std = std

    paired_dataloader = build_paired_dataloader(config, adata_train)

    # Determine logger for embed phase
    if wandb_logger is not None:
        phase_logger = wandb_logger
    elif config.use_wandb:
        config_dict = OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
        phase_logger = WandbLogger(project=project, name="embed", log_model=True)
        wandb.init(config=config_dict, project=project, reinit=True)
    else:
        phase_logger = False

    trainer = build_trainer(config, phase_logger, "embed")

    if phase_logger:
        phase_logger.watch(embed_model, log="gradients", log_freq=50)

    trainer.fit(model=embed_model, train_dataloaders=paired_dataloader)

    if phase_logger:
        wandb.unwatch(embed_model)
    if wandb_logger is None and config.use_wandb:
        wandb.finish()

    embed_model.eval()
    freeze_params(embed_model)

    return ae_model, disc_model, metric_model, embed_model


def train(config, project, wandb_logger=None):
    """Main entry point for GAGA pipeline. Same signature as run_model.train()."""

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    adata = process_data(
        pc_dim=config.pc_dim,
        t0_index=config.t0_index,
        t1_index=config.t1_index,
        data=config.dataset,
        use_paga=config.use_paga,
        tissue=config.tissue,
    )
    config.num_classes = adata.obs['cell_type'].nunique()

    timepoints = sorted(adata.obs['timepoint'].unique().tolist())

    t0, t1 = timepoints[config.t0_index], timepoints[config.t1_index]

    adata = adata[(adata.obs['timepoint'] >= t0) & (adata.obs['timepoint'] <= t1)]
    adata_train = adata[adata.obs['timepoint'].isin([t0, t1])]

    ae_model, disc_model, metric_model, embed_model = run_gaga_model(
        config=config,
        project=project,
        adata_train=adata_train,
        timepoints=timepoints,
        wandb_logger=wandb_logger,
    )

    remove_all_forward_hooks(ae_model)
    remove_all_forward_hooks(metric_model)
    remove_all_forward_hooks(embed_model)

    # Evaluation: W1 distance to held-out timepoints
    w1_scores = []
    for index in range(config.t0_index + 1, config.t1_index):
        t = timepoints[index]
        w1 = predict(embed_model, adata, t0, t, t1, num_traj=6000, library="pot")
        w1_scores.append(w1)
    w1_scores = torch.tensor(w1_scores)

    return ae_model, disc_model, metric_model, embed_model, w1_scores
