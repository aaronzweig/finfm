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
)
from models.metric_models import MetricNetGAGA, FinslerGAGA
from models.embed_models import EmbedNetTrainBase
from models.modules import SimpleEmbedNet, SinNet
from datasets.process import process_data, extract_paired_dataset
from datasets.dataset import ShufflingDataset, ShufflingOTDataset
from utils.frozen import freeze_params
from utils.callback import BestModelCallback
from eval.eval import predict
from scripts.run_model import (
    build_embed, build_paired_dataloader, build_trainer,
    build_classifier, build_singleton_dataloader,
    remove_all_forward_hooks,
)


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
    # When an external logger is provided (sweep mode) don't reuse it here — each pl.Trainer
    # resets its global_step to 0, causing step-number collisions in the wandb dashboard.
    if wandb_logger is not None:
        logger = False
    elif config.use_wandb:
        logger = WandbLogger(project="gaga", name="autoencoder")
    else:
        logger = False
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
    """Generate ambient-space negatives and train discriminator on raw X_pca.

    The discriminator (a ModelBase subclass) normalizes internally, so we feed it raw
    X_pca during training.  At inference inside MetricNetGAGA._fn, x is already
    normalized and the discriminator is called with normalize=False.
    Negatives are generated in raw ambient space to match the X_pca scale.
    """
    # Generate negative samples (off-manifold) in raw ambient space
    noise_levels = list(config.gaga.noise_levels)
    negatives = neg_sample_additive(X_pca, noise_levels, seed=config.seed)

    # Optional rejection sampling
    if config.gaga.sampling_rejection:
        rejected_mask = sampling_rejection(
            X_pca,
            negatives,
            method=config.gaga.sampling_rejection_method,
            k=config.gaga.sampling_rejection_k,
            threshold=config.gaga.sampling_rejection_threshold,
        )
        negatives = negatives[~rejected_mask]

    # Build discriminator dataloader (raw X_pca; discriminator normalizes internally)
    pos = torch.tensor(X_pca, dtype=torch.float32)
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

    # Discriminator input dim = ambient pc_dim (not latent dim)
    disc_model = Discriminator(
        config=config,
        in_dim=config.pc_dim,
        layer_widths=config.gaga.disc_hidden_dims,
        disc_lr=config.gaga.disc_lr,
        disc_weight_decay=config.gaga.disc_weight_decay,
        dropout=config.gaga.disc_dropout,
        use_spectral_norm=config.gaga.disc_use_spectral_norm,
    )
    # Share normalization with AE so disc sees the same normalized space at inference
    disc_model.mean = ae_model.mean
    disc_model.std = ae_model.std

    # Same logger fix as train_autoencoder: avoid step-counter collisions in sweep mode
    if wandb_logger is not None:
        logger = False
    elif config.use_wandb:
        logger = WandbLogger(project="gaga", name="discriminator")
    else:
        logger = False
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

from scipy.spatial.distance import pdist, squareform
from scipy import sparse as sp
import phate

def calculate_distances(X, seed=42, knn=5, t='auto', n_components=3):
    # if X is sparse, convert to dense
    if sp.issparse(X):
        X = X.toarray()
        
    phate_op = phate.PHATE(random_state=seed, t=t, n_components=n_components, knn=knn, verbose=0)
    phate_data = phate_op.fit_transform(X)
    
    dists = squareform(pdist(phate_op.diff_potential))

    return dists

# --- Main pipeline ---

def run_gaga_model(config, project, adata_train, timepoints, tree, wandb_logger=None):
    """Run the full GAGA pipeline: autoencoder -> discriminator -> (classifier) -> embed."""

    X_pca = adata_train.obsm['X_pca']

    distances = calculate_distances(adata_train.obsm['X_pca'])

    # Phase 0: Compute normalization stats (ModelBase-style)
    X_tensor = torch.from_numpy(X_pca).float()
    mean = torch.mean(X_tensor, dim=0, keepdim=True)
    std = torch.max(torch.std(X_tensor, dim=0)) * np.sqrt(config.pc_dim)

    # Get distance matrix
    if issparse(distances):
        distances = distances.toarray()
    distances = np.asarray(distances, dtype=np.float32)
    # Use only upper-triangular (off-diagonal) values — squareform includes n zeros on
    # the diagonal that would deflate the std and over-scale the normalized distances.
    n = len(distances)
    dist_std = np.std(distances[np.triu_indices(n, k=1)])

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

    # Phase 2.5 (optional): Classifier for Finsler mode
    classifier_model = None
    if config.finsler.use:
        print("Phase 2.5: Training classifier for Finsler metric...")
        classifier_model = build_classifier(config)
        classifier_model.mean = mean
        classifier_model.std = std

        singleton_dataloader = build_singleton_dataloader(config, adata_train)

        # Determine logger — same pattern as run_model.py
        if wandb_logger is not None:
            phase_logger = wandb_logger
        elif config.use_wandb:
            config_dict = OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
            phase_logger = WandbLogger(project=project, name="classifier", log_model=True)
            wandb.init(config=config_dict, project=project, reinit=True)
        else:
            phase_logger = False

        trainer = build_trainer(config, phase_logger, "classifier")
        if phase_logger:
            phase_logger.watch(classifier_model, log="gradients", log_freq=50)
        trainer.fit(model=classifier_model, train_dataloaders=singleton_dataloader)
        if phase_logger:
            wandb.unwatch(classifier_model)
        if wandb_logger is None and config.use_wandb:
            wandb.finish()

        classifier_model.eval()
        freeze_params(classifier_model)

    # Phase 3: Build metric model — branch on finsler flag
    print("Phase 3: Training embed model...")
    if config.finsler.use:
        metric_model = FinslerGAGA(
            config=config,
            encoder=ae_model,
            discriminator=disc_model,
            disc_factor=config.gaga.disc_factor,
            classifier_model=classifier_model,
            tree=torch.from_numpy(tree).float(),
            temp=config.finsler.temp,
            lamb=config.finsler.lamb,
        )
    else:
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

    return classifier_model, ae_model, disc_model, metric_model, embed_model


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
    tree = adata.uns['tree']

    t0, t1 = timepoints[config.t0_index], timepoints[config.t1_index]

    adata = adata[(adata.obs['timepoint'] >= t0) & (adata.obs['timepoint'] <= t1)]
    adata_train = adata[adata.obs['timepoint'].isin([t0, t1])]

    classifier_model, ae_model, disc_model, metric_model, embed_model = run_gaga_model(
        config=config,
        project=project,
        adata_train=adata_train,
        timepoints=timepoints,
        tree=tree,
        wandb_logger=wandb_logger,
    )

    remove_all_forward_hooks(ae_model)
    remove_all_forward_hooks(metric_model)
    remove_all_forward_hooks(embed_model)
    if classifier_model is not None:
        remove_all_forward_hooks(classifier_model)

    # Evaluation: W1 distance to held-out timepoints
    w1_scores = []
    for index in range(config.t0_index + 1, config.t1_index):
        t = timepoints[index]
        w1 = predict(embed_model, adata, t0, t, t1, num_traj=6000, library="pot")
        w1_scores.append(w1)
    w1_scores = torch.tensor(w1_scores)

    return classifier_model, ae_model, disc_model, metric_model, embed_model, w1_scores
