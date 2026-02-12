import os
import scanpy as sc
import torch
from datasets.dataset import *
from utils.preprocess import *
from utils.lineage import *

def process_data(pc_dim, data="zebrafish", use_paga=False, paga_threshold=0.0):
    path = "data"
    if data == "zebrafish": #constrain to neural cells in control
        suffix = "pairwise_hvg.h5ad"
        filename = os.path.join(path, suffix)
        adata = load_data(filename)

        subset = (adata.obs['gene_target'] == 'ctrl-inj') & (adata.obs['tissue'] == "Central Nervous System")
        adata = adata[subset]
        sc.tl.pca(adata, n_comps = pc_dim, mask_var = None)

        adata.uns['std'] = np.ones((1,pc_dim))
        adata.obs['cell_type'] = adata.obs['cell_type_broad']

        if use_paga:
            print("using paga")
            adj = run_paga_tree(adata, 'cell_type', threshold=paga_threshold, use_tree=True)
        else:
            adj = ZEBRAFISH_NEURAL_ADJACENCY
        adata = incorporate_tree(adata, adj, 'cell_type')

    elif data == "cite":
        suffix = "cite.h5ad"
        filename = os.path.join(path, suffix)
        adata = sc.read(filename)
        adata.obs['gene_target'] = ['ctrl-inj'] * adata.shape[0]
        adata.obs['timepoint'] = adata.obs['day']

        adata = incorporate_tree(adata, CITE_ADJACENCY, 'cell_type')

        #TODO: the wrong donor!???? But we downloaded it from https://data.mendeley.com/datasets/hhny5ff7yj/1

    elif data == "multi":
        pass

    elif data == "cite_gaga":
        filename = "/home/azweig/projects/finfm/benchmark/flow_matching_minimal/data/cite_100.h5ad"
        adata = sc.read(filename)
        adata.obs['gene_target'] = ['ctrl-inj'] * adata.shape[0]
        adata.obs['timepoint'] = adata.obs['day']

        adata = incorporate_tree(adata, CITE_ADJACENCY, 'cell_type')

    elif data == "mouse":
        import scipy.io as sio
        import pandas as pd

        cache_file = os.path.join("atlas", "mouse_preprocessed.h5ad")
        if os.path.exists(cache_file):
            print("Loading cached preprocessed mouse data...")
            adata = sc.read(cache_file)
        else:
            print("Loading raw mouse atlas data (this may take a few minutes)...")
            counts = sio.mmread(os.path.join("atlas", "raw_counts.mtx")).T.tocsr()
            meta = pd.read_csv(os.path.join("atlas", "meta.csv"))
            genes = pd.read_csv(os.path.join("atlas", "genes.tsv"), sep='\t', header=None,
                                names=['ensembl_id', 'gene_symbol'])

            adata = sc.AnnData(X=counts)
            adata.obs = meta.reset_index(drop=True)
            adata.obs.index = adata.obs['cell'].values
            adata.var_names = genes['gene_symbol'].values
            adata.var_names_make_unique()

            is_doublet = adata.obs['doublet'].astype(str).str.upper() == 'TRUE'
            is_stripped = adata.obs['stripped'].astype(str).str.upper() == 'TRUE'
            has_celltype = adata.obs['celltype'].notna()
            valid_stage = adata.obs['stage'] != 'mixed_gastrulation'
            mask = ~is_doublet & ~is_stripped & has_celltype & valid_stage
            adata = adata[mask].copy()

            sc.pp.filter_cells(adata, min_genes=100)
            sc.pp.filter_genes(adata, min_cells=3)
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)
            sc.pp.highly_variable_genes(adata, n_top_genes=2000)
            adata = adata[:, adata.var['highly_variable']].copy()

            adata.write(cache_file)
            print(f"Saved preprocessed data to {cache_file}")

        adata.obs['cell_type'] = adata.obs['celltype'].astype(str)

        stage_to_num = {
            'E6.5': 6.5, 'E6.75': 6.75, 'E7.0': 7.0, 'E7.25': 7.25,
            'E7.5': 7.5, 'E7.75': 7.75, 'E8.0': 8.0, 'E8.25': 8.25, 'E8.5': 8.5
        }
        adata.obs['timepoint'] = adata.obs['stage'].map(stage_to_num).astype(float)

        # PCA
        sc.tl.pca(adata, n_comps=pc_dim, mask_var=None)
        adata.uns['std'] = np.ones((1, pc_dim))

        # Tree: always use PAGA for mouse (no manual adjacency defined)
        if use_paga:
            print("using paga")
        adj = run_paga_tree(adata, 'cell_type', threshold=paga_threshold, use_tree=True)
        adata = incorporate_tree(adata, adj, 'cell_type')

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

