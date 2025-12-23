import numpy as np
import torch
import scipy
import wandb
from pytorch_lightning.loggers import WandbLogger
import torch
import os
import pickle
import torch
from torch.utils.data import Dataset, Sampler, DataLoader, TensorDataset
import pytorch_lightning as pl
import random
#from utils.ot import *

class Config(dict):
    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")
    def __setattr__(self, key, value):
        self[key] = value
    def __delattr__(self, item):
        try:
            del self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")

def sample(X, batch_size):
    replace = batch_size > X.shape[0]
    indices = np.random.choice(X.shape[0], batch_size, replace=replace)
    return X[indices]



class ShufflingKNNDataset(Dataset):
    """Even more optimized version with additional tricks"""
    
    def __init__(self, X, adj):

        self.X = X
        self.adj_csr = adj.tocsr()
        
        self.neighbor_starts = []
        self.neighbor_counts = []
        self.all_neighbors = []
        
        for i in range(self.X.shape[0]):
            start_idx = self.adj_csr.indptr[i]
            end_idx = self.adj_csr.indptr[i + 1]
            neighbors = self.adj_csr.indices[start_idx:end_idx]
            neighbors = neighbors[neighbors != i]  # Remove self
            
            self.neighbor_starts.append(len(self.all_neighbors))
            self.neighbor_counts.append(len(neighbors))
            self.all_neighbors.extend(neighbors)
        
        # Convert to numpy for faster indexing
        self.neighbor_starts = np.array(self.neighbor_starts)
        self.neighbor_counts = np.array(self.neighbor_counts)
        self.all_neighbors = np.array(self.all_neighbors)
        
        # Pre-generate random numbers to avoid repeated calls
        self.batch_size = 1000
        self.random_idx = 0
        self.random_t_values = torch.rand(self.batch_size)
        self.random_neighbor_choices = torch.randint(0, 1000, (self.batch_size,))
    
    def _refresh_random_values(self):
        """Refresh random values when batch is exhausted"""
        self.random_t_values = torch.rand(self.batch_size)
        self.random_neighbor_choices = torch.randint(0, 1000, (self.batch_size,))
        self.random_idx = 0
    
    def __getitem__(self, index):
        # Get neighbor info
        start = self.neighbor_starts[index]
        count = self.neighbor_counts[index]
        
        if count == 0:
            return (self.X[index],)
        
        # Use pre-generated random values
        if self.random_idx >= self.batch_size:
            self._refresh_random_values()
        
        # Get neighbor
        neighbor_offset = self.random_neighbor_choices[self.random_idx].item() % count
        neighbor_idx = self.all_neighbors[start + neighbor_offset]
        
        # Get interpolation parameter
        t = self.random_t_values[self.random_idx].item()
        self.random_idx += 1
        
        # Interpolate
        x = self.X[index]
        y = self.X[neighbor_idx]
        return ((1 - t) * x + t * y,)
    
    def __len__(self):
        return self.X.shape[0]







# class ShufflingKNNDataset(Dataset):
#     def __init__(self, X, adj):

#         self.X = X
#         self.adj = adj.tocsr()

#     def _get_neighbors(self, row_idx):
#         start = self.adj.indptr[row_idx]
#         end = self.adj.indptr[row_idx + 1]
#         cols = self.adj.indices[start:end]
#         return cols

#     #TODO: bias towards original data?  Sample from simplex?
#     def __getitem__(self, index):

#         x = self.X[index]
#         y = self.X[np.random.choice(self._get_neighbors(index))]
#         t = np.random.rand()
#         z = (1-t) * x + t * y
#         return (z,)
        
#     def __len__(self):
#         return self.X.shape[0]



class ShufflingDataset(Dataset):
    def __init__(self, obj_list, batch_size, conditions):

        self.obj_list = obj_list
        self.batch_size = batch_size
        self.conditions = conditions

    def __getitem__(self, index):
        obj = self.obj_list[index]
        X0, X1, t0, t1, gene_target = obj

        x0 = sample(X0, self.batch_size)
        x1 = sample(X1, self.batch_size)
        
        return x0, x1, torch.tensor(t0), torch.tensor(t1), self.conditions[gene_target]

    def __len__(self):
        return len(self.obj_list)


class ShufflingOTDataset(Dataset):
    def __init__(self, obj_list, batch_size, conditions, update_epoch_rate = 50):

        self.obj_list = obj_list
        self.batch_size = batch_size
        self.conditions = conditions
        self.update_epoch_rate = update_epoch_rate
        self.ot_list = None

    def __getitem__(self, index):
        obj = self.ot_list[index]
        X, Y, t0, t1, gene_target = obj

        replace = self.batch_size > X.shape[0]
        indices = np.random.choice(X.shape[0], self.batch_size, replace=replace)

        #Share indices because we have prior OT
        x = X[indices]
        y = Y[indices]
        
        return x, y, torch.tensor(t0), torch.tensor(t1), self.conditions[gene_target]

    def __len__(self):
        return len(self.obj_list)

    def update(self, epoch, embed_net):
        if epoch % self.update_epoch_rate == 0:
            self.recompute_ot_samples(embed_net)

    @torch.no_grad()
    def embed(self, X, Y, embed_net):
        embed_net.eval()
        embedded_X = []
        embedded_Y = []
        device = next(embed_net.parameters()).device

        xloader = DataLoader(TensorDataset(X),
                            batch_size=2048, #TODO: Hard-coded?
                            shuffle=False)
        yloader = DataLoader(TensorDataset(Y),
                            batch_size=2048, #TODO: Hard-coded?
                            shuffle=False)

        with torch.no_grad():
            for xb in xloader:
                xb = xb[0].to(device, non_blocking=True)
                ex = embed_net(xb)
                embedded_X.append(ex.cpu())

            for yb in yloader:
                yb = yb[0].to(device, non_blocking=True)
                ey = embed_net(yb)
                embedded_Y.append(ey.cpu())

        # Concatenate all results
        embedded_X = torch.cat(embedded_X, dim=0)
        embedded_Y = torch.cat(embedded_Y, dim=0)

        embed_net.train()
        
        return embedded_X, embedded_Y

    def recompute_ot_samples(self, embed_net):
        self.ot_list = []
        for obj in self.obj_list:
            X, Y, t0, t1, gene_target = obj
            X_emb, Y_emb = self.embed(X, Y, embed_net)
            i, j = sample_from_coupling(X_emb, Y_emb, n_samples=X.shape[0]+Y.shape[0], indices=True)
            self.ot_list.append((X[i], Y[j], t0, t1, gene_target))
            
        