import anndata as ad
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def sample_quarter(size, inp, sigma=0.05, d=2):
    i, o = inp #angle, offset
    f = lambda x: (np.cos(x), np.sin(x))
    z = np.stack(f(np.pi/2 * i + np.pi/2 * np.random.uniform(size=size)), axis = 1)
    if d > 2:
        x = np.zeros((z.shape[0], d))
        x[:,:2] = z
        z = x
    ep = sigma * np.random.normal(size=(size, d))
    offset = np.pad(np.array(o), (0, d-2), 'constant')[None, :]
    return z + ep + offset

def sample_s(size, d):
    sizes = [size//6] * 6
    angles = [0, 1, 2, 0, 3, 2]
    offsets = [[0,1], [0,1], [0,1], [0, -1], [0, -1], [0, -1]]
    inputs = list(zip(angles, offsets))

    x = np.concatenate([sample_quarter(sizes[i], inputs[i], d=d) for i in range(len(sizes))], axis = 0)
    y = np.zeros(x.shape[0])
    t = np.repeat([0,1,2,3,4,5], size//6)
    tree = np.zeros((1,1))
    
    return x, y, t, tree

def sample_points(size, d):
    sizes = [size//3] * 3

    x0 = np.zeros((sizes[0], d))
    x1 = np.zeros((sizes[1], d))
    x2 = np.zeros((sizes[2], d))
    x1[:,0] = 1
    x1[:,1] = 1
    x2[:,0] = 2
    x = np.concatenate([x0, x1, x2], axis = 0)

    y = np.zeros(x.shape[0])
    t = np.repeat([0,1,2], size//3)
    tree = np.zeros((1,1))
    
    return x, y, t, tree

def sample_diamond(size, d):

    x = 0.2 * np.random.normal(size=(size, d))
    x0, x1, x2, x3 = tuple(np.array_split(x, indices_or_sections = 4, axis = 0))
    x0[:,0] -= 1.0
    x1[:,0] += 1.0
    x2[:,1] -= 1.0
    x3[:,1] += 1.0
    x = np.concatenate([x0, x1, x2, x3], axis=0)

    y0 = np.zeros(x0.shape[0])
    y1 = np.zeros_like(x1.shape[0]) + 1
    y2 = np.zeros_like(x2.shape[0]) + 2
    y3 = np.zeros_like(x3.shape[0]) + 3
    y = np.concatenate([y0, y1, y2, y3], axis=0)

    t0 = np.zeros_like(x0.shape[0])
    t1 = np.zeros_like(x1.shape[0])
    t2 = np.zeros_like(x2.shape[0]) + 1
    t3 = np.zeros_like(x3.shape[0]) + 1
    t = np.concatenate([t0, t1, t2, t3], axis=0)

    tree = np.zeros((4,4))
    tree[0,2] = 1
    tree[1,3] = 1
    
    return x, y, t, tree

def process_fake_data(size, d, sample_fn, normalize=False):
    x, y, t, tree = sample_fn(size, d)

    adata = ad.AnnData(x)
    adata.obs['timepoint'] = pd.Categorical(t)
    adata.obs['gene_target'] = pd.Categorical(["ctrl-inj"] * x.shape[0])
    adata.obs['cell_type'] = pd.Categorical(y)
    adata.obs['cell_type_one_hot'] = adata.obs['cell_type']

    #TODO: handle LabelEncoder for cell types properly
    
    adata.obsm['X_pca'] = adata.X
    adata.varm['PCs'] = np.zeros((d, d))
    adata.uns['std'] = np.ones((1,d))
    adata.uns['tree'] = tree

    if normalize:
        adata.uns['std'] = np.std(adata.obsm['X_pca'], axis=0, keepdims=True)
        adata.obsm['X_pca'] /= adata.uns['std']

    return adata