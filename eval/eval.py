import numpy as np
import torch

from geomloss import SamplesLoss
from datasets.process import *

def ot_dist(x, y, a = None, b = None, p = 1, library = "geomloss"):

    if p not in [1,2]:
        raise NotImplementedError(f"p = {p} not allowed.")

    if library == "geomloss":
        return geomloss_ot_dist(x, y, a, b, p)
    elif library == "pot":
        return pot_ot_dist(x, y, a, b, p)
    else:
        raise NotImplementedError(f"Library {library} not known.")

def geomloss_ot_dist(x, y, a = None, b = None, p=2):
    x = x.cuda()
    y = y.cuda()
    loss = SamplesLoss(loss="sinkhorn", p=p, blur=.05)
    if a is None:
        a = torch.ones(x.shape[0]) / x.shape[0]
    a = a.cuda()
    if b is None:
        b = torch.ones(y.shape[0]) / y.shape[0]
    b = b.cuda()
    if p == 2:
        return torch.sqrt(2 * loss(a, x, b, y)).item()
    if p == 1:
        return loss(a, x, b, y).item()
    return None

# def ot_dist(x, y):
#     x = x.cuda()
#     y = y.cuda()
#     loss = SamplesLoss(loss="sinkhorn", p=2, blur=.05)
#     return torch.sqrt(2 * loss(x, y)).item()

import ot
def pot_ot_dist(X, Y, a = None, b = None, p = 1):
    if a is None:
        a = torch.ones(X.shape[0]) / X.shape[0]
    if b is None:
        b = torch.ones(Y.shape[0]) / Y.shape[0]
    if p == 2:
        M = ot.dist(X, Y, metric='sqeuclidean')
        dist = ot.emd2(a, b, M) ** 0.5
    elif p == 1:
        M = ot.dist(X, Y, metric='euclidean')
        dist = ot.emd2(a, b, M)
    # ot_epsilon = 0.05
    # dist = ot.sinkhorn2(a, b, M, ot_epsilon, method='sinkhorn_log') ** (1/p)
    return dist

def predict(embed_model, adata, t, p=1, num_traj=2000, batch_size=512, library="geomloss"):
    
    timepoints = sorted(adata.obs['timepoint'].unique().tolist())
    index = timepoints.index(t)
        
    t0 = timepoints[index-1]
    t1 = timepoints[index+1]

    adata_obs = adata[adata.obs['timepoint'].isin([t0, t1])]

    dataset = extract_paired_dataset(adata_obs)
    train_dataset = ShufflingDataset(dataset, batch_size) #TODO: replace with ShufflingOTDataset when that actually works
    train_dataloader = DataLoader(train_dataset, batch_size = 1, shuffle=True)

    samples = []
    total = 0
    while total < num_traj:
        batch = next(iter(train_dataloader))
        x = embed_model.sample_geodesic_time(batch, t)
        samples.append(x)
        total += x.shape[0]
    samples = torch.cat(samples, dim=0)[:num_traj]

    true = torch.tensor(adata[adata.obs['timepoint'] == t].obsm['X_pca']).to(embed_model.device)
    
    emp_dist = ot_dist(samples, true, None, None, p=p, library=library)

    return emp_dist