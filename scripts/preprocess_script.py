import numpy as np
import torch
import scanpy as sc
import anndata as ad
import pandas as pd

import os 
import sys
sys.path.append(os.path.abspath(".."))
sys.path.append(os.path.abspath("."))

from utils.preprocess import *

if __name__ == "__main__":

    path = "/home/azweig/projects/zebrafish/data"
    suffix = "1k_hvg.h5ad"
    filename = os.path.join(path, suffix)

    adata = import_data()
    adata = preprocess(adata, n_top_genes=1000, exclude_highly_expressed=False)
    adata.write(filename)