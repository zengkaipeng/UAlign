import pandas
import os
from utils.chemistry_parse import (
    get_reaction_core, get_bond_info, BOND_FLOAT_TO_TYPE,
    BOND_FLOAT_TO_IDX
)
from model import EditDataset
from utils.graph_utils import smiles2graph
from utils.chemistry_parse import clear_map_number
import random
import numpy as np
import torch
from tqdm import tqdm
import rdkit
from rdkit import Chem


def create_sparse_dataset(
    reacts, prods, rxn_class=None, kekulize=False,
    return_amap=False, verbose=True
):
    amaps, graphs, nodes, edge_types, ret = [], [], [], [], []
    for idx, prod in enumerate(tqdm(prods) if verbose else prods):
        ret.append(clear_map_number(reacts[idx]))
        x, y = get_reaction_core(reacts[idx], prod, kekulize=kekulize)
        graph, amap = smiles2graph(prod, with_amap=True, kekulize=kekulize)
        graphs.append(graph)
        nodes.append([amap[t] for t in x])
        es = []
        for edgs in y:
            src, dst, _, _ = edgs.split(':')
            src, dst = int(src), int(dst)
            es.append((amap[src], amap[dst]))
        edge_types.append(es)

        amaps.append(amap)
    dataset = EditDataset(graphs, nodes, edge_types, ret, rxn_class)
    return (dataset, amaps) if return_amap else dataset


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


if __name__ == '__main__':
    pass
