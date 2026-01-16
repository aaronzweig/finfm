import numpy as np
import torch

from geomloss import SamplesLoss
from datasets.process import *

def ot_dist(x, y, a = None, b = None, p=2):
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

# import ot
# def ot_dist(X, Y, ot_epsilon = 0.1):
#     a = torch.ones(X.shape[0]) / X.shape[0]
#     b = torch.ones(Y.shape[0]) / Y.shape[0]
#     M = ot.dist(X, Y)
#     ot_epsilon = 0.1
#     dist = ot.sinkhorn2(a, b, M, ot_epsilon, method='sinkhorn_log')
#     return dist

def predict(embed_model, adata, t, p=1, num_traj=2000, batch_size=512):
    
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
    
    emp_dist = ot_dist(samples, true, None, None, p=p)

    return emp_dist

# def predict(model, adata, value, conditions, num_traj, t, p=2, steps=2000):
#     adata_local = adata[(adata.obs['gene_target'] == value) | (adata.obs['gene_target'] == 'ctrl-inj')]    
#     adata_ctrl = adata_local[adata_local.obs['gene_target'] == 'ctrl-inj']
#     adata_per = adata_local[adata_local.obs['gene_target'] == value]
    
#     timepoints = sorted(adata_per.obs['timepoint'].unique().tolist())
#     index = timepoints.index(t)
#     if index == 0:
#         print("failure")
#         return -1
        
#     t0 = timepoints[index-1]
#     t1 = t    
        
#     c = conditions[value]
#     c = c.unsqueeze(0).repeat(num_traj, 1)

#     X0 = torch.tensor(adata_per[adata_per.obs['timepoint'] == timepoints[index-1]].obsm['X_pca']).to(model.device)
#     X1 = torch.tensor(adata_per[adata_per.obs['timepoint'] == timepoints[index]].obsm['X_pca']).to(model.device)

#     samples, weights = model.sample_and_weight(X0, c, t0, t1, num_traj, steps=steps)
#     samples = samples.detach().cpu()
#     if weights is not None:
#         weights = weights.detach().cpu()
    
#     emp_dist = ot_dist(samples, torch.from_numpy(np.array(X1)), weights, None, p=p)

#     return emp_dist

# def predict_ctrl(model, adata, value, conditions, num_traj, t, p=2):
#     adata_local = adata[(adata.obs['gene_target'] == value) | (adata.obs['gene_target'] == 'ctrl-inj')]    
#     adata_ctrl = adata_local[adata_local.obs['gene_target'] == 'ctrl-inj']
#     adata_per = adata_local[adata_local.obs['gene_target'] == value]

#     X = adata_per[adata_per.obs['timepoint'] == t].obsm['X_pca']
#     Y = adata_ctrl[adata_ctrl.obs['timepoint'] == t].obsm['X_pca']
    
#     indices = np.random.choice(Y.shape[0], num_traj, replace=False)
#     Y = Y[indices]
#     ctrl_dist = ot_dist(torch.from_numpy(np.array(Y)), torch.from_numpy(np.array(X)), p=p)

#     return ctrl_dist

# def predict_exact(model, adata, value, conditions, num_traj, t, p=2):
#     adata_local = adata[(adata.obs['gene_target'] == value) | (adata.obs['gene_target'] == 'ctrl-inj')]    
#     adata_ctrl = adata_local[adata_local.obs['gene_target'] == 'ctrl-inj']
#     adata_per = adata_local[adata_local.obs['gene_target'] == value]

#     X = adata_per[adata_per.obs['timepoint'] == t].obsm['X_pca']
    
#     indices = np.random.choice(X.shape[0], num_traj, replace=False)
#     Y = X[indices]
#     true_dist = ot_dist(torch.from_numpy(np.array(Y)), torch.from_numpy(np.array(X)), p=p)

#     return true_dist