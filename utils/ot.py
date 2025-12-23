import torch
import numpy as np
from geomloss import SamplesLoss
from pykeops.torch import LazyTensor


#https://github.com/getkeops/keops/issues/372
def uniform_lazy_tensor_2d(dim0, dim1, device) -> LazyTensor:
    """Initialize a 2D lazy tensor with uniform random values."""
    rand_x = LazyTensor(torch.rand((dim0, 1, 1), device=device))
    rand_y = LazyTensor(torch.rand((1, dim1, 1), device=device))

    rand_xy = (rand_x * 12.9898 + rand_y * 78.233).sin() * 43758.5453123
    rand_xy_floor = (rand_xy - 0.5).round()
    rand_xy_fract = rand_xy - rand_xy_floor
    rand_xy_clamp = rand_xy_fract.clamp(0, 1)

    return rand_xy_clamp

@torch.no_grad()
def sample_from_coupling_keops_batched(
    X, Y, f, g, p, eps, n_samples,
    i_batch=1024,           # tune: how many source rows to handle at once
):
    """
    Batched (over source rows) KeOps-based sampler that avoids N x M allocations.
    - X: (N,d) torch tensor on device
    - Y: (M,d) torch tensor on device
    - f: (N,) dual potentials
    - g: (M,) dual potentials
    - p: exponent (usually 2)
    - eps: temperature (blur**p)
    - n_samples: total samples to draw (draws source indices uniformly here)
    - i_batch: number of distinct source rows processed per iteration
    Returns (i_idx, j_idx) both long tensors of length n_samples.
    """
    device = X.device
    N, M = X.shape[0], Y.shape[0]

    i_idx = torch.randint(0, N, (n_samples,), device=device)

    pairs_j = []

    for start in range(0, n_samples, i_batch):
        end = min(start + i_batch, n_samples)
        idx_batch = i_idx[torch.arange(start,end)]      # (B,)
        B = idx_batch.shape[0]

        Xb = X[idx_batch]                    # (B,d)
        fb = f[idx_batch]                    # (B,)

        X_i = LazyTensor(Xb[:, None, :])     # (B, 1, d)
        Y_j = LazyTensor(Y[None, :, :])      # (1, M, d)

        C = ((X_i - Y_j).abs() ** p).sum(-1) / p

        F = LazyTensor(fb[:, None, None])     # (B,1,1)
        G = LazyTensor(g[None, :, None])      # (1,M,1)

        logits = (F + G - C) / eps

        U = uniform_lazy_tensor_2d(B, M, device)
        gumbel = -(-U.log()).log()
        noisy = logits + gumbel

        # argmax over M for each row
        j_batch = noisy.argmax(dim=1)    # (B,1) long on device
        
        pairs_j.append(j_batch.squeeze(-1))  # store chosen target indices (B,)

    j_idx = torch.cat(pairs_j, dim=0)

    return i_idx, j_idx


def sample_from_coupling(X, Y, n_samples, blur = 0.05, p = 2, scaling = 0.9, indices=False):

    eps = blur ** (1/p)
    
    loss_fn = SamplesLoss("sinkhorn", p=p, blur=blur, scaling=scaling, debias=False, potentials=True)
    alpha, beta = loss_fn(X, Y)
    alpha, beta = alpha.flatten(), beta.flatten()
    
    i, j = sample_from_coupling_keops_batched(
        X, Y, alpha, beta, p=p,
        eps=eps,
        n_samples=n_samples
    )

    if indices:
        return i, j
    return X[i], Y[j]    

# X = torch.concat([torch.rand(100000, 2) + torch.tensor([[0.0, 5.0]]), torch.rand(20000, 2) + torch.tensor([[0.0, -5.0]])], axis=0)
# Y = torch.rand(200000, 2) + torch.tensor([[10.0, 0.0]])

# X_, Y_ = sample_from_coupling(X, Y, 200000)
    
# Z = 0.5 * X_ + 0.5 * Y_

# plt.scatter(X[:,0], X[:,1])
# plt.scatter(Z[:,0], Z[:,1])
# plt.scatter(Y[:,0], Y[:,1])
# plt.show()