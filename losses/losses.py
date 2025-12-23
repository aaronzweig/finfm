import numpy as np
import torch
import torch.nn as nn
from torch.func import jvp, vmap, jacrev

def dsm_score_estimation_mixture(
    scorenet,
    samples,
    sigma,
    pre_scorenet,
    beta,
    pre_beta,
    alpha=1.0,
):

    # naive dsm
    noise = torch.randn_like(samples)
    perturbed_samples = samples + noise * sigma
    target = - (perturbed_samples - samples) / (sigma ** 2)
    target /= beta
    
    scores = scorenet(perturbed_samples)
    target = target.view(target.shape[0], -1)
    scores = scores.view(scores.shape[0], -1)
    loss = ((scores - target) ** 2).sum(dim=-1)

    if pre_scorenet:
        # matching prev score
        prev_scores = pre_scorenet(perturbed_samples).detach()
        # prev_score_loss = ((scores - (beta/pre_beta) * prev_scores) ** 2).sum(dim=-1)
        prev_score_loss = (torch.abs(scores - (beta/pre_beta) * prev_scores)).sum(dim=-1)
        loss = alpha * loss + (1 - alpha) * prev_score_loss

    return loss.mean()
