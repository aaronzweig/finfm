import numpy as np
import torch
import scanpy as sc
import anndata as ad
import os
import pandas as pd

def import_zebra_data():
    
    path = "/home/azweig/projects/zebrafish/data"
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

    return adata

def preprocess(adata, min_genes=100, min_cells=3, n_top_genes=2000):
    adata = import_zebrafish_data()
    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_genes(adata, min_cells=min_cells)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    adata.var["gene_short_name"] = adata.var["gene_short_name"].astype(str)
    adata.var_names = adata.var["gene_short_name"].values
    adata.var_names_make_unique()

    targets = list(adata.obs['gene_target'].unique())
    hvg = pd.Series(False, index=adata.var_names)
    
    for gene_target in targets:
        adata_paired = adata[(adata.obs['gene_target']=='ctrl-inj') | (adata.obs['gene_target']==gene_target)]
        sc.pp.highly_variable_genes(adata_paired, n_top_genes=n_top_genes)
        hvg = hvg | adata_paired.var['highly_variable']
        
    adata.var['highly_variable'] = hvg
    return adata

#TODO: this omits all the weird interventions?  Like *-mut interventions?
def load_data(filename):
    adata = sc.read(filename)
    targets = list(adata.obs['gene_target'].unique())
    hvg = adata.var['highly_variable']
    perturbed = adata.var['gene_short_name'].isin(targets)
    adata = adata[:,(hvg) | (perturbed)]
    return adata