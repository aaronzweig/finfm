import matplotlib.pyplot as plt
import numpy as np
import torch
import scanpy as sc
import anndata as ad
import umap


def restrict_adata(adata, value):
    return adata[(adata.obs['gene_target'] == value) | (adata.obs['gene_target'] == 'ctrl-inj')]

def save_umap(adata_local):
    
    X = adata_local.obsm['X_pca']
    umap_model = umap.UMAP().fit(X) #pca of ctrl adata
    
    adata_local.obsm["X_umap"] = umap_model.embedding_
    return adata_local, umap_model

def plot_samples(model, adata, value, conditions, num_traj, t):
    adata_local = restrict_adata(adata, value)
    adata_local, umap_model = save_umap(adata_local)
    adata_ctrl = adata_local[adata_local.obs['gene_target'] == 'ctrl-inj']
    adata_per = adata_local[adata_local.obs['gene_target'] == value]
    
    timepoints = sorted(adata_per.obs['timepoint'].unique().tolist())
    index = timepoints.index(t)
    if index == 0:
        print("failure")
        return -1
        
    t0 = timepoints[index-1]
    t1 = t    
        
    c = conditions[value]
    c = c.unsqueeze(0).repeat(num_traj, 1)
    
    X0 = adata_per[adata_per.obs['timepoint'] == timepoints[index-1]].obsm['X_pca']
    X1 = adata_per[adata_per.obs['timepoint'] == timepoints[index]].obsm['X_pca']

    samples = model.sample(X0, c, t0, t1, num_traj).detach().cpu()
    samples_umap = umap_model.transform(samples)

    fig, (ax0, ax1) = plt.subplots(2, figsize=(16, 16))

    sc.pl.umap(
        adata_local[adata_local.obs['timepoint'] == t],
        color="gene_target",
        # Setting a smaller point size to get prevent overlap
        size=2,
        ax=ax0,
        show=False
    )
    
    sc.pl.umap(
        adata_ctrl[adata_ctrl.obs['timepoint'] == t],
        # color="tissue",
        # Setting a smaller point size to get prevent overlap
        size=2,
        ax=ax1,
        show=False
    )
    
    ax1.scatter(samples_umap[:, 0], samples_umap[:, 1], marker=".", linestyle="-", alpha=0.7, lw=1.5, color='#ff7f0e')
    
    plt.show()


def plot_trajectories(model, adata, value, conditions, num_traj):
    adata_local = restrict_adata(adata, value)
    adata_local, umap_model = save_umap(adata_local)

    adata_ctrl = adata_local[adata_local.obs['gene_target'] == 'ctrl-inj']
    adata_per = adata_local[adata_local.obs['gene_target'] == value]

    timepoints = sorted(adata_per.obs['timepoint'].unique().tolist())
    t0, t1 = min(timepoints), max(timepoints)
    
    c = conditions[value]
    c = c.unsqueeze(0).repeat(num_traj, 1)
    
    X0 = adata_per[adata_per.obs['timepoint'] == min(timepoints)].obsm['X_pca']

    traj = model.sample_traj(X0, c, t0, t1, num_traj)

    trajectories = [traj[:,i].numpy() for i in range(traj.shape[1])]
    trajectories_umap = [umap_model.transform(traj) for traj in trajectories]

    fig, (ax0, ax1, ax2, ax3) = plt.subplots(4, figsize=(16, 16))

    sc.pl.umap(
        adata_local,
        color="tissue",
        # Setting a smaller point size to get prevent overlap
        size=2,
        ax=ax0,
        show=False,
    )
    
    sc.pl.umap(
        adata_local,
        color="gene_target",
        # Setting a smaller point size to get prevent overlap
        size=2,
        ax=ax1,
        show=False,
    )

    sc.pl.umap(adata_per, color="timepoint", ax=ax2, show=False)
    
    sc.pl.umap(adata_per, color="timepoint", ax=ax3, show=False)
    
    for traj in trajectories_umap:
        ax3.plot(traj[:, 0], traj[:, 1], marker=".", linestyle="-", alpha=0.7, lw=1.5)

    plt.tight_layout()
    plt.show()

    return trajectories

def plot_vector_field(model, adata, value, conditions, num_traj):
    adata_local = restrict_adata(adata, value)
    adata_local, umap_model = save_umap(adata_local)

    adata_ctrl = adata_local[adata_local.obs['gene_target'] == 'ctrl-inj']
    adata_per = adata_local[adata_local.obs['gene_target'] == value]

    timepoints = sorted(adata_per.obs['timepoint'].unique().tolist())
    t0, t1 = min(timepoints), max(timepoints)
    
    c = conditions[value]
    c = c.unsqueeze(0).repeat(num_traj, 1)
    
    X0 = adata_per[adata_per.obs['timepoint'] == min(timepoints)].obsm['X_pca']

    traj = model.sample_traj(X0, c, t0, t1, num_traj)
    
    trajectories = [traj[:,i].numpy() for i in range(traj.shape[1])]
    trajectories_umap = [umap_model.transform(traj) for traj in trajectories]

    fig, (ax1, ax2) = plt.subplots(2, figsize=(16, 16))
    
    sc.pl.umap(
        adata_per,
        color="tissue",
        # Setting a smaller point size to get prevent overlap
        size=2,
        ax=ax1,
        show=False
    )
    
    sc.pl.umap(adata_per, color="timepoint", ax=ax2, show=False)
    
    for traj in trajectories_umap:
        # print(traj[0], traj[1])
        diffs = np.diff(traj, axis=0)
        ax2.quiver(traj[0, 0], traj[0, 1], diffs[0, 0], diffs[0, 1],
                   angles='xy', scale_units='xy', color='r', 
                   linewidth=0.2, alpha=0.5)
    
    plt.show()
    