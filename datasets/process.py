import os
import scanpy as sc
import torch
from datasets.dataset import *
from utils.preprocess import *




def process_data(pc_dim = None, data="zebrafish"):
    path = "/home/mingxuanzhang/zebrafish"
    if not os.path.isdir(path):
        path = "/home/azweig/projects/zebrafish/data"
    if data == "zebrafish":
        suffix = "pairwise_hvg.h5ad"
        filename = os.path.join(path, suffix)
        adata = load_data(filename)
        sc.tl.pca(adata, n_comps = pc_dim, mask_var = None) #because load_data already filters for hvg + perturbed genes
    elif data == "cite":
        suffix = "cite.h5ad"
        filename = os.path.join(path, suffix)
        adata = sc.read(filename)
        adata.obs['gene_target'] = ['ctrl-inj'] * adata.shape[0]
        adata.obs['timepoint'] = adata.obs['day']
        if pc_dim != 100:
            sc.tl.pca(adata, n_comps = pc_dim)
        #TODO: the wrong donor!???? But we downloaded it from https://data.mendeley.com/datasets/hhny5ff7yj/1
    elif data == "EB":
        suffix = "EB.h5ad"
        filename = os.path.join(path, suffix)
        adata = sc.read(filename)
        adata.obs['gene_target'] = ['ctrl-inj'] * adata.shape[0]
        adata.obs["timepoint"] = adata.obs["timepoint"].cat.codes + 1
        sc.tl.pca(adata, n_comps = pc_dim)

    key = 'gene_target'
    values = []
    for c in list(adata.obs['gene_target'].unique()):
        # if c in adata.var_names.tolist(): #would mute gene_targets that aren't exactly one gene
        values.append(c)

    return adata, values

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

