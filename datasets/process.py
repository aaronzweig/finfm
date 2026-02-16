import os
import numpy as np
import scanpy as sc
import torch
from datasets.dataset import *
from utils.preprocess import *
from utils.lineage import *

def process_data(pc_dim, t0_index, t1_index, data="zebrafish", use_paga=False, paga_threshold=0.0, tissue="Central Nervous System"):
    path = "data"
    if data == "zebrafish": 
        suffix = "pairwise_hvg.h5ad"
        filename = os.path.join(path, suffix)
        adata = load_data(filename)

        subset = (adata.obs['gene_target'] == 'ctrl-inj') & (adata.obs['tissue'] == tissue)
        adata = adata[subset]
        adata.obs['cell_type'] = adata.obs['cell_type_broad']
        # sc.tl.pca(adata, n_comps=pc_dim, mask_var=None)

        # Fit PCA on t0/t1 subset, apply transform to all data
        timepoints = sorted(adata.obs['timepoint'].unique().tolist())
        t0, t1 = timepoints[t0_index], timepoints[t1_index]
        subset = adata[adata.obs['timepoint'].isin([t0, t1])].copy()
        sc.tl.pca(subset, n_comps=pc_dim, mask_var=None)

        mean = np.mean(subset.X, axis=0)
        PCs = subset.varm['PCs']
        adata.obsm['X_pca'] = (adata.X - mean) @ PCs
        adata.varm['PCs'] = PCs
        adata.uns['pca'] = subset.uns['pca']

        if tissue == "Central Nervous System" and not use_paga:
            adj = ZEBRAFISH_NEURAL_ADJACENCY
        elif tissue == "Pharyngeal Arch" and not use_paga:
            adj = ARCH_NEURAL_ADJACENCY
        else:
            print("using paga")
            adj = run_paga_tree(adata, 'cell_type', threshold=paga_threshold, use_tree=True)
        adata = incorporate_tree(adata, adj, 'cell_type')

    # elif data == "cite":
    #     suffix = "cite.h5ad"
    #     filename = os.path.join(path, suffix)
    #     adata = sc.read(filename)
    #     adata.obs['gene_target'] = ['ctrl-inj'] * adata.shape[0]
    #     adata.obs['timepoint'] = adata.obs['day']

    #     adata = incorporate_tree(adata, CITE_ADJACENCY, 'cell_type')

    #     #TODO: the wrong donor!???? But we downloaded it from https://data.mendeley.com/datasets/hhny5ff7yj/1

    # elif data == "celegans":
    #     suffix = "celegans.h5ad"
    #     filename = os.path.join(path, suffix)
    #     adata = sc.read(filename)
    #     adata.obs['gene_target'] = ['ctrl-inj'] * adata.shape[0]
    #     dic = {'300_minutes': 300, '400_minutes': 400, '500_minutes': 500}
    #     adata.obs['timepoint'] = [dic[x] for x in adata.obs['time_point'].tolist()]
    #     sc.tl.pca(adata, n_comps = pc_dim)

    #     adj = run_paga_tree(adata, 'cell_type', threshold=paga_threshold, use_tree=True)
    #     adata = incorporate_tree(adata, adj, 'cell_type')

    # elif data == "cite_gaga":
    #     filename = "/home/azweig/projects/finfm/benchmark/flow_matching_minimal/data/cite_100.h5ad"
    #     adata = sc.read(filename)
    #     adata.obs['gene_target'] = ['ctrl-inj'] * adata.shape[0]
    #     adata.obs['timepoint'] = adata.obs['day']

    #     adata = incorporate_tree(adata, CITE_ADJACENCY, 'cell_type')
    
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

