import torch
from contextlib import contextmanager


@contextmanager
def frozen_params(module):
    old_flags = [p.requires_grad for p in module.parameters()]
    for p in module.parameters():
        p.requires_grad_(False)
    yield
    for p, old in zip(module.parameters(), old_flags):
        p.requires_grad_(old)

def freeze_params(module):
    for p in module.parameters():
        p.requires_grad_(False)