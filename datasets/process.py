import os
import numpy as np
import scanpy as sc
import torch
import pandas as pd
from datasets.dataset import *
from utils.preprocess import *
from utils.lineage import *
import networkx as nx
import scipy.io as sio




def process_data(pc_dim, t0_index, t1_index, data="zebrafish", 
                 use_paga=False, paga_threshold=0.2, 
                 tissue="Central Nervous System"):
    path = "data"
    if data == "zebrafish": 
        suffix = "zebra_preprocessed.h5ad"
        filename = os.path.join(path, suffix)
        if os.path.exists(filename):
            adata = load_data(filename)
        else:
            print("Loading raw zebrafish atlas data...")
            mtx_suffix = "zscape_perturb_full_raw_counts.mtx"
            cell_suffix = "zscape_perturb_full_cell_metadata.csv"
            gene_suffix = "zscape_perturb_full_gene_metadata.csv"

            mtx_filename = os.path.join(path, mtx_suffix)
            cell_filename = os.path.join(path, cell_suffix)
            gene_filename = os.path.join(path, gene_suffix)

            adata = sc.read_mtx(mtx_filename)
            cell_metadata = pd.read_csv(cell_filename)
            gene_metadata = pd.read_csv(gene_filename)

            adata = adata.transpose()
            adata.obs = cell_metadata
            adata.var = gene_metadata

            adata.var["gene_short_name"] = adata.var["gene_short_name"].astype(str)
            adata.var_names = adata.var["gene_short_name"].values
            adata.var_names_make_unique()

            adata = adata[adata.obs['gene_target'] == 'ctrl-inj']

            sc.pp.filter_cells(adata, min_genes=100)
            sc.pp.filter_genes(adata, min_cells=3)
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)
            sc.pp.highly_variable_genes(adata, n_top_genes=2000)

            adata = adata[:, adata.var['highly_variable']].copy()

            adata.write(filename)
            print(f"Saved preprocessed data to {filename}")

        subset = adata.obs['tissue'] == tissue
        adata = adata[subset]
        adata.obs['cell_type'] = adata.obs['cell_type_broad']
        if tissue == "Blood":
            adata.obs['cell_type'] = adata.obs['cell_type_broad']
        sc.tl.pca(adata, n_comps=pc_dim, mask_var=None)

        if tissue == "Central Nervous System" and not use_paga:
            adj = ZEBRAFISH_NEURAL_ADJACENCY
        elif tissue == "Pharyngeal Arch" and not use_paga:
            adj = ARCH_NEURAL_ADJACENCY
        else:
            print("using paga")
            cns_root_nodes=['neural progenitor (telencephalon/diencephalon)',
                            'neural progenitor (MHB)',
                            'neural progenitor (hindbrain)',
                            'neural progenitor (hindbrain R7/8)',
                            'posterior spinal cord progenitors'
                            ]
            arch_root_nodes = ['cranial muscle (progenitor)',
                               'head mesenchyme (maybe ventral, hand2+)']
            blood_root_nodes = []
            root_node = ""
            if tissue == "Central Nervous System":
                root_node = cns_root_nodes
            elif tissue == "Pharyngeal Arch":
                root_node = arch_root_nodes
            elif tissue == "Blood":
                root_node = blood_root_nodes
            adj = run_paga_tree(adata, 'cell_type', threshold=paga_threshold, root_node=root_node)
        adata = incorporate_tree(adata, adj, 'cell_type')

    elif data == "cite":
        suffix = "cite.h5ad"
        filename = os.path.join(path, suffix)
        adata = sc.read(filename)
        adata.obs['gene_target'] = ['ctrl-inj'] * adata.shape[0]
        adata.obs['timepoint'] = adata.obs['day']

        adata = incorporate_tree(adata, CITE_ADJACENCY, 'cell_type')

    elif data == "mouse":
        BLOOD = [ "Blood progenitors 1",
                   "Blood progenitors 2",
                    "Haematoendothelial progenitors",
                    "Erythroid1",
                    "Erythroid2",
                    "Erythroid3"]
        
        BRAIN = [
                    "Spinal cord",
                    "Caudal neurectoderm",
                    "Caudal epiblast",
                    "NMP",
                    ]
        cache_file = os.path.join(path, "atlas", "mouse_preprocessed.h5ad")
        if os.path.exists(cache_file):
            print("Loading cached preprocessed mouse data...")
            adata = sc.read(cache_file)
        else:
            print("Loading raw mouse atlas data...")

            counts = sio.mmread(os.path.join(path, "atlas", "raw_counts.mtx")).T.tocsr()
            meta = pd.read_csv(os.path.join(path, "atlas", "meta.csv"))
            genes = pd.read_csv(os.path.join(path, "atlas", "genes.tsv"), sep='\t', header=None,
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
        
        # SUBSET TO TISSUE
        if tissue == "brain":
            cell_list = BRAIN
            root_node = ["NMP", "Caudal epiblast"]
        elif tissue == "blood":
            cell_list = BLOOD
            root_node ="Haematoendothelial progenitors"
        else:
            cell_list = None
            root_node = None
        adata = adata[adata.obs['cell_type'].isin(cell_list)].copy()
        adata.obs['cell_type'] = adata.obs['cell_type'].astype('category').cat.remove_unused_categories()

        # PCA
        sc.tl.pca(adata, n_comps=pc_dim, mask_var=None)
        adata.uns['std'] = np.ones((1, pc_dim))

        # Tree: always use PAGA for mouse 
        if use_paga:
            print("using paga")
        
        adj = run_paga_tree(adata, 'cell_type', threshold=paga_threshold, root_node=root_node)    
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

