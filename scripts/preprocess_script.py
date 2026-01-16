import numpy as np
import torch
import scanpy as sc
import anndata as ad
import pandas as pd

import os 
import sys

from utils.preprocess import *

#TODO: doesn't actually run yet, need to fix module stuff

if __name__ == "__main__":

    filename = "1k_hvg.h5ad"

    adata = import_zebrafish_data()
    adata = preprocess(adata, n_top_genes=1000, exclude_highly_expressed=False)
    adata.write(filename)