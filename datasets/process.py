import os
import scanpy as sc
import torch
from datasets.dataset import *
from utils.preprocess import *
from utils.lineage import *

def process_data(pc_dim, data="zebrafish"):
    path = "data"
    if data == "zebrafish": #constrain to neural cells in control
        suffix = "pairwise_hvg.h5ad"
        filename = os.path.join(path, suffix)
        adata = load_data(filename)

        subset = (adata.obs['gene_target'] == 'ctrl-inj') | (adata.obs['tissue'] == "Central Nervous System")
        adata = adata[subset]
        sc.tl.pca(adata, n_comps = pc_dim, mask_var = None) #because load_data already filters for hvg + perturbed genes

        adata.uns['std'] = np.ones((1,pc_dim))
        adata.obs['cell_type'] = adata.obs['cell_type_broad']
        incorporate_tree(adata, ZEBRAFISH_NEURAL_ADJACENCY, 'cell_type')

    elif data == "cite":
        suffix = "cite.h5ad"
        filename = os.path.join(path, suffix)
        adata = sc.read(filename)
        adata.obs['gene_target'] = ['ctrl-inj'] * adata.shape[0]
        adata.obs['timepoint'] = adata.obs['day']

        incorporate_tree(adata, CITE_ADJACENCY, 'cell_type')

        #TODO: the wrong donor!???? But we downloaded it from https://data.mendeley.com/datasets/hhny5ff7yj/1

    elif data == "multi":
        pass
    
    return adata

def extract_paired_dataset(adata, use_rep='X_pca'):
    
    dataset = []
    seen_timepoints = sorted(adata.obs['timepoint'].unique().tolist())
    for i in range(len(seen_timepoints)-1):
        t0 = seen_timepoints[i]
        t1 = seen_timepoints[i+1]
        adata0 = adata[adata.obs['timepoint'] == t0]
        adata1 = adata[adata.obs['timepoint'] == t1]
        X0 = torch.from_numpy(adata0.obsm[use_rep]).float()
        X1 = torch.from_numpy(adata1.obsm[use_rep]).float()
        obj = (X0, X1, t0, t1)
        dataset.append(obj)

    return dataset

def extract_singleton_dataset(adata, use_rep='X_pca'):

    dataset = torch.from_numpy(adata.obsm[use_rep]).float()
    y = torch.from_numpy(adata.obs['cell_type_one_hot'].to_numpy()).long()
    return dataset, y

