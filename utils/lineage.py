ZEBRAFISH_NEURAL_ADJACENCY = {
    'neural progenitor (telencephalon/diencephalon)': [
        'differentiating neuron 1',
        'differentiating neuron 2'
    ],
    'neural progenitor (MHB)': [
        'differentiating neuron 1',
        'differentiating neuron 2'
    ],
    'neural progenitor (hindbrain)': [
        'differentiating neuron (hindbrain)'
    ],
    'neural progenitor (hindbrain R7/8)': [
        'differentiating neuron (hindbrain)',
        'differentiating neuron 1',
        'differentiating neuron 2',
        'motor neuron'
    ],
    'posterior spinal cord progenitors': [
        'differentiating neuron (hindbrain)',
        'dorsal spinal cord neuron',
        'motor neuron'
    ],
    'differentiating neuron (hindbrain)': [
        'hypophysis/locus coeruleus',
        'neuron (+ spinal cord)'
    ],
    'differentiating neuron 1': [
        'neuron (dopaminergic)',
        'neurons (gabaergic, glutamatergic; contains Purkinje)',
        'neurons (gabaergic, glutamatergic)',
        'neuron (telencephalon, glutamatergic)'
    ],
    'differentiating neuron 2': [
        'neuron (dopaminergic)',
        'neurons (gabaergic, glutamatergic; contains Purkinje)',
        'neurons (gabaergic, glutamatergic)',
        'neuron (telencephalon, glutamatergic)'
    ],
    'neurons (differentiating, contains peripheral)': [
        'neurons (gabaergic, glutamatergic; contains Purkinje)',
        'neurons (gabaergic, glutamatergic)',
        'neuron (telencephalon, glutamatergic)'
    ],
}

CITE_ADJACENCY = {
    'HSC': ['NeuP', 'EryP', 'MasP', 'MkP', 'MoP', 'BP']
}

import itertools
import numpy as np
from sklearn.preprocessing import LabelEncoder
def incorporate_tree(adata, adj, cell_type_key):
    cell_types = adata.obs[cell_type_key].unique().tolist()
    tree_cell_types = list(adj.keys()) + list(itertools.chain(*adj.values()))
    cell_types = list(set(cell_types + tree_cell_types))

    stoi = {name: i for i, name in enumerate(cell_types)}
    itos = {i: name for i, name in enumerate(cell_types)}

    N = len(cell_types)
    tree = np.zeros((N, N))

    for u, neighbors in adj.items():
        for v in neighbors:
            tree[stoi[u], stoi[v]] = 1

    adata.uns['stoi'] = stoi
    adata.uns['itos'] = itos
    adata.uns['tree'] = tree
    adata.obs['cell_type_one_hot'] = [stoi[x] for x in adata.obs[cell_type_key]]


