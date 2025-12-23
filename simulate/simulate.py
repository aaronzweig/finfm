import anndata as ad
import numpy as np
import pandas as pd

def process_fake_data(sizes, inputs, d, sample_fn, normalize=False):
    xs = []
    fs = []
    for i in range(len(sizes)):
        f = lambda s=sizes[i], inp=inputs[i]: sample_fn(s, inp, d=d)
        x = f()
        fs.append(f)
        xs.append(x)

    x = np.concatenate(xs, axis = 0)
    adata = ad.AnnData(x)
    times = [[i] * s for i, s in enumerate(sizes)]
    adata.obs['timepoint'] = pd.Categorical(sum(times, []))
    adata.obs['gene_target'] = pd.Categorical(["ctrl-inj"] * sum(sizes))

    
    adata.obsm['X_pca'] = adata.X
    adata.obsm['X_pca_raw'] = adata.X.copy()
    adata.varm['PCs'] = np.zeros((d, d))
    adata.uns['std'] = np.ones((1,d))

    if normalize:
        adata.uns['std'] = np.std(adata.obsm['X_pca'], axis=0, keepdims=True)
        adata.obsm['X_pca'] /= adata.uns['std']

    return adata, ['ctrl-inj'], fs