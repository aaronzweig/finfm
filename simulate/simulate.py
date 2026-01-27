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

def sample_s_imbalanced(size, d):
    sizes = [size//20, size//10, size, size//20, size, size//20]
    angles = [0, 1, 2, 0, 3, 2]
    offsets = [[0,1], [0,1], [0,1], [0, -1], [0, -1], [0, -1]]
    inputs = list(zip(angles, offsets))

    x = np.concatenate([sample_quarter(sizes[i], inputs[i], d=d) for i in range(len(sizes))], axis = 0)
    y = np.zeros(x.shape[0])
    t = [[i] * sizes[i] for i in range(len(sizes))]
    t = np.concatenate(t, axis = 0)
    tree = np.zeros((1,1))
    
    return x, y, t, tree

def sample_omega(size, d):
    sizes = [size//6] * 6
    sigma = 0.05

    x0 = np.zeros((sizes[0], d))
    x0[:,0] = np.random.uniform(0, 2, size=sizes[0])
    x0[:,1] = -1.0
    x0 += sigma * np.random.normal(size=(sizes[0], d))
    x1 = sample_quarter(sizes[1], (3, [0,0]), d=d)
    x2 = sample_quarter(sizes[2], (0, [0,0]), d=d)
    x3 = sample_quarter(sizes[3], (1, [0,0]), d=d)
    x4 = sample_quarter(sizes[4], (2, [0,0]), d=d)
    x5 = np.zeros((sizes[5], d))
    x5[:,0] = np.random.uniform(-2, 0, size=sizes[5])
    x5[:,1] = -1.0
    x5 += sigma * np.random.normal(size=(sizes[5], d))

    x = np.concatenate([x0, x1, x2, x3, x4, x5], axis = 0)
    y = np.repeat([0,1,2,3,4,5], size//6)
    t = np.repeat([0,0,0,1,1,1], size//6)
    tree = np.zeros((6,6))
    tree[0,1] = 1
    tree[1,2] = 1
    tree[2,3] = 1
    tree[3,4] = 1
    tree[4,5] = 1
    return x, y, t, tree

def sample_wedge(size, d):
    r = np.sqrt(np.random.uniform(0, 1, size=size))
    theta = np.pi / 2 * np.random.uniform(0, 1, size=size)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.stack([x,y], axis=1)

def sample_disk(size, d):
    sizes = [size//4] * 4
    sigma = 0.05
    assert d == 2

    x0 = sample_wedge(sizes[0], d)
    x1 = sample_wedge(sizes[1], d) * np.array([[-1, 1]])
    x2 = sample_wedge(sizes[2], d) * np.array([[-1, -1]])
    x3 = sample_wedge(sizes[3], d) * np.array([[1, -1]])

    x = np.concatenate([x0, x1, x2, x3], axis = 0)
    y = np.repeat([0,1,2,3], size//4)
    t = np.repeat([0,0,1,1], size//4)
    tree = np.zeros((4,4))
    tree[0,1] = 1
    tree[1,2] = 1
    tree[2,3] = 1
    return x, y, t, tree

def sample_growth(size, d):
    sizes = [size//5] * 5
    sigma = 0.1
    assert d == 2

    x0 = sigma * np.random.normal(size=(sizes[0], d)) + np.array([[0, 0]])
    x1 = sigma * np.random.normal(size=(sizes[1], d)) + np.array([[0.5, 0]])
    x2 = sigma * np.random.normal(size=(sizes[2], d)) + np.array([[0, 0]])
    x3 = sigma * np.random.normal(size=(sizes[3], d)) + np.array([[0, -0.5]])
    x4 = sigma * np.random.normal(size=(sizes[4], d)) + np.array([[0.5, -0.5]])

    x = np.concatenate([x0, x1, x2, x3, x4], axis = 0)
    y = np.repeat([0,1,0,2,3], size//5)
    t = np.repeat([0,0,1,1,1], size//5)
    tree = np.zeros((4,4))
    tree[1,2] = 1
    tree[1,3] = 1
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

    sizes = [size//4] * 4

    x0 = 0.1 * np.random.normal(size=(sizes[0], d))
    x1 = 0.1 * np.random.normal(size=(sizes[1], d))
    x2 = 0.1 * np.random.normal(size=(sizes[2], d))
    x3 = 0.1 * np.random.normal(size=(sizes[3], d))

    x0[:,0] -= 1.0
    x1[:,0] += 1.0
    x2[:,1] -= 1.0
    x3[:,1] += 1.0
    x = np.concatenate([x0, x1, x2, x3], axis=0)

    y0 = np.zeros(x0.shape[0])
    y1 = np.zeros(x1.shape[0]) + 1
    y2 = np.zeros(x2.shape[0]) + 2
    y3 = np.zeros(x3.shape[0]) + 3
    y = np.concatenate([y0, y1, y2, y3], axis=0)

    t0 = np.zeros(x0.shape[0])
    t1 = np.zeros(x1.shape[0])
    t2 = np.zeros(x2.shape[0]) + 1
    t3 = np.zeros(x3.shape[0]) + 1
    t = np.concatenate([t0, t1, t2, t3], axis=0)

    tree = np.zeros((4,4))
    tree[0,2] = 1
    tree[1,3] = 1
    
    return x, y, t, tree

from torchdyn.datasets import generate_moons
import math
import torch
def sample_moons(size, d):

    assert d == 2, "Not implemented high dim moons yet"

    var = 0.1
    scale = 5
    m = torch.distributions.multivariate_normal.MultivariateNormal(
        torch.zeros(d), math.sqrt(var) * torch.eye(d)
    )
    centers = [
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1),
        (1.0 / np.sqrt(2), 1.0 / np.sqrt(2)),
        (1.0 / np.sqrt(2), -1.0 / np.sqrt(2)),
        (-1.0 / np.sqrt(2), 1.0 / np.sqrt(2)),
        (-1.0 / np.sqrt(2), -1.0 / np.sqrt(2)),
    ]
    centers = torch.tensor(centers) * scale
    noise = m.sample((size,))
    multi = torch.multinomial(torch.ones(8), size, replacement=True)
    data = []
    for i in range(size):
        data.append(centers[multi[i]] + noise[i])
    x0 = torch.stack(data).numpy()

    x1, _ = generate_moons(size, noise=0.2)
    x1 = x1.numpy()
    x1 = x1 * 3 - 1

    x = np.concatenate([x0, x1], axis=0)

    y = np.zeros(x.shape[0])
    t0 = np.zeros(x0.shape[0])
    t1 = np.zeros(x1.shape[0]) + 1
    t = np.concatenate([t0, t1], axis=0)

    tree = np.zeros((1,1))
    
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