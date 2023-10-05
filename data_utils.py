import pandas
import os
from utils.chemistry_parse import (
    get_reaction_core, get_bond_info, BOND_FLOAT_TO_TYPE,
    BOND_FLOAT_TO_IDX, get_modified_atoms_bonds
)
from utils.graph_utils import smiles2graph
from Dataset import OverallDataset
from Dataset import BinaryEditDataset
import random
import torch
import numpy as np
from tqdm import tqdm
from typing import Any, Dict, List, Tuple, Optional, Union


def get_lap_pos_encoding(
    matrix: np.ndarray, dim: int, num_nodes: Optional[int] = None
) -> torch.Tensor:
    if num_nodes is None:
        num_nodes = matrix.shape[0]
    N = np.diag(matrix.sum(axis=-1).clip(1) ** -0.5)
    L = np.eye(num_nodes) - np.matmul(np.matmul(N, matrix), N)
    EigVal, EigVec = np.linalg.eig(L)
    EigVec = EigVec[:, EigVal.argsort()]
    t_result = EigVec[:, 1: dim + 1]
    result, rdim = torch.zeros(num_nodes, dim), t_result.shape[1]
    if rdim > 0:
        result[:, -rdim:] = torch.from_numpy(t_result.real).float()
    return result


def add_pos_enc(graph, method='none', **kwargs):
    assert method in ['none', 'Lap'], f'Invalid node pos enc {method}'
    if method == 'Lap':
        matrix = np.zeros((graph['num_nodes'], graph['num_nodes']))
        matrix[graph['edge_index'][0], graph['edge_index'][1]] = 1
        matrix[graph['edge_index'][1], graph['edge_index'][0]] = 1
        graph['lap_pos_enc'] = get_lap_pos_encoding(
            matrix=matrix, num_nodes=graph['num_nodes'], **kwargs
        )
        return graph
    else:
        return graph


def create_edit_dataset(
    reacts, prods, rxn_class=None, kekulize=False,
    verbose=True, pos_enc='none', **kwargs
):
    amaps, graphs, nodes, edges = [], [], [], []
    for idx, prod in enumerate(tqdm(prods) if verbose else prods):
        x, y = get_modified_atoms_bonds(reacts[idx], prod, kekulize=kekulize)
        graph, amap = smiles2graph(prod, with_amap=True, kekulize=kekulize)
        graph = add_pos_enc(graph, method=pos_enc, **kwargs)
        graphs.append(graph)
        amaps.append(amap)
        nodes.append([amap[t] for t in x])
        edges.append([(amap[i], amap[j]) for i, j in y])

    return BinaryEditDataset(graphs, nodes, edges, rxn_class=rxn_class)


def extend_amap(amap, node_list):
    result, max_node = {}, max(amap.values())
    for node in node_list:
        if node not in amap:
            result[node] = max_node + 1
            max_node += 1
        else:
            result[node] = amap[node]
    return result


def create_overall_dataset(
    reacts, prods, rxn_class=None, kekulize=False,
    verbose=True, pos_enc='none', **kwargs
):
    graphs, nodes, edges = [], [], []
    node_types, edge_types = [], []
    for idx, prod in enumerate(tqdm(prods) if verbose else prods):
        # encoder_part
        x, y = get_modified_atoms_bonds(reacts[idx], prod, kekulize)
        encoder_graph, prod_amap = smiles2graph(
            prod, with_amap=True, kekulize=kekulize
        )
        encoder_graph = add_pos_enc(encoder_graph, method=pos_enc, **kwargs)
        graphs.append(graph)
        nodes.append([amap[t] for t in x])
        edges.append([(amap[i], amap[j]) for i, j in y])

        node_type = get_node_types(reacts)
        extend_amap = extend_amap(prod_amap, node_type.keys())
        edge_type = get_edge_types(reacts, kekulize=kekulize)

        real_n_types = {extend_amap[k]: v for k, v in node_type.items()}
        real_e_types = {
            (extend_amap[x], extend_amap[y]): v
            for (x, y), v in edge_types.items()
        }
        node_types.append(real_n_types)
        edge_types.append(real_e_types)
        return OverallDataset(
            graphs=graphs, activate_nodes=nodes, changed_edges=edges,
            decoder_node_type=real_n_types, decoder_edge_type=real_e_types,
            rxn_class=rxn_class
        )


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
        answer &= all(t <= x[0] for t in x[1:])
    return answer


if __name__ == '__main__':
    print(load_data('../data/USPTO-50K', 'train'))
