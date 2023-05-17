import pandas
import os
from utils.chemistry_parse import (
    get_reaction_core, get_bond_info, BOND_FLOAT_TO_TYPE,
    BOND_FLOAT_TO_IDX
)
from model import EditDataset
from utils.graph_utils import smiles2graph
import random
import torch
import numpy as np


def create_sparse_dataset(
    reacts, prods, rxn_class=None, kekulize=False,
    return_amap=False
):
    amaps, graphs, nodes, edge_types = [], [], [], []
    for idx, prod in enumerate(prods):
        x, y, z = get_reaction_core(reacts[idx], prod, kekulize=kekulize)
        graph, amap = smiles2graph(prod, with_amap=True, kekulize=kekulize)
        graphs.append(graph)
        nodes.append([amap[t] for t in x])
        edge_types.append({
            (amap[i], amap[j]): BOND_FLOAT_TO_IDX[v[0]]
            for (i, j), v in z.items() if i in amap and j in amap
        })
        amaps.append(amap)
    dataset = EditDataset(graphs, nodes, edge_types)
    return (dataset, amap) if return_amap else amap


def load_data(data_dir, part):
    df_train = pandas.read_csv(
        os.path.join(data_dir, f'canonicalized_raw_{part}.csv')
    )
    rxn_class, reacts, prods = [], [], []
    for idx, resu in enumerate(df_train['reactants>reagents>production']):
        rxn_class.append(df_train['class'][idx])
        rea, prd = resu.strip().split('>>')
        reacts.append(rea)
        prods.append(prd)
    return reacts, prods, rxn_class


def fix_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)


def check_early_stop(*args):
    answer = True
    for x in args:
        answer &= all(t < x[0] for t in x[1:])
    return answer


if __name__ == '__main__':
    print(load_data('../data/USPTO-50K', 'train'))
