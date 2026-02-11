import itertools
import numpy as np
import scanpy as sc


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
import scanpy as sc

def run_paga_tree(adata, cell_type_key, threshold=0.0, use_tree=True):
    sc.pp.neighbors(adata, use_rep="X_pca")
    sc.tl.paga(adata, groups=cell_type_key)
    categories = adata.obs[cell_type_key].cat.categories.tolist()
    if use_tree:
        conn = adata.uns['paga']['connectivities_tree'].toarray()
    else:
        conn = adata.uns['paga']['connectivities'].toarray()
        conn[conn < threshold] = 0

    # Build adjacency dictionary
    adj = {}
    n = len(categories)
    for i in range(n):
        neighbors = []
        for j in range(n):
            if conn[i, j] > 0:
                neighbors.append(categories[j])
        if neighbors:
            adj[categories[i]] = neighbors

    return adj

from sklearn.preprocessing import LabelEncoder
#TODO: we currently prune to only have cell_types in the tree
def incorporate_tree(adata, adj, cell_type_key):
    # cell_types = adata.obs[cell_type_key].unique().tolist()
    # tree_cell_types = list(adj.keys()) + list(itertools.chain(*adj.values()))
    # cell_types = list(set(cell_types + tree_cell_types))

    tree_cell_types = list(adj.keys()) + list(itertools.chain(*adj.values()))
    tree_cell_types = list(set(tree_cell_types))
    adata = adata[adata.obs[cell_type_key].isin(tree_cell_types)]
    cell_types = tree_cell_types

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
    return adata

