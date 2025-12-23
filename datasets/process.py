import os
import scanpy as sc
import torch
from datasets.dataset import *




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

def get_condition_vector(adata, value, default='ctrl-inj'):
    dim = adata.varm['PCs'].shape[1]
    names = adata.var_names.tolist()
    
    if value == default:
        return torch.zeros(dim,)
    #PC loadings are useless, use random embeddings
    z = torch.randn(dim,)
    return z / torch.linalg.norm(z)
    

def extract_dataset(adata, values, default='ctrl-inj', use_rep='X_pca'):

    conditions = {value: get_condition_vector(adata, value, default) for value in values}
    
    dataset = []
    for gene_target in values:
        z = adata[adata.obs['gene_target'] == gene_target]
        seen_timepoints = sorted(z.obs['timepoint'].unique().tolist())
        for i in range(len(seen_timepoints)-1):
            t0 = seen_timepoints[i]
            t1 = seen_timepoints[i+1]
            adata0 = z[z.obs['timepoint'] == t0]
            adata1 = z[z.obs['timepoint'] == t1]
            X0 = torch.from_numpy(adata0.obsm[use_rep]).float()
            X1 = torch.from_numpy(adata1.obsm[use_rep]).float()
            obj = (X0, X1, t0, t1, gene_target)
            dataset.append(obj)

    return conditions, dataset

def extract_score_dataset(adata, values, default='ctrl-inj', use_rep='X_pca', n_neighbors=10, resolution=0.1):

    conditions = {value: get_condition_vector(adata, value, default) for value in values}

    adata_train = adata[adata.obs['gene_target'].isin(values)]
    dataset = torch.from_numpy(adata_train.obsm[use_rep]).float()

    sc.pp.neighbors(adata_train, n_neighbors=n_neighbors, use_rep=use_rep)
    sc.tl.leiden(adata_train, resolution=resolution)
    
    return conditions, dataset, torch.from_numpy(np.array(adata_train.obs['leiden'].astype(int).tolist()))

