import pandas
import os
from utils.chemistry_parse import (
    get_reaction_core, get_bond_info, BOND_FLOAT_TO_TYPE,
    BOND_FLOAT_TO_IDX
)
from backBone import EditDataset


def create_sparse_dataset(
    reacts, prods, rxn_class=None, kekulize=False,
    return_amap=False
):
    amaps, graphs, nodes, edge_types = [], [], [], []
    for idx, reac in enumerate(reacts):
        x, y, z = get_reaction_core(reac, prods[idx], kekulize=kekulize)
        graph, amap = smiles2graph(prod, with_amap=True, kekulize=kekulize)
        graphs.append(graph)
        nodes.append([amap[t] for t in x])
        edge_types.append({
            (amap[i], amap[j]): BOND_FLOAT_TO_IDX[v[0]]
            for (i, j), v in z.items() if i in amap and j in amap
        })
        amaps.append(amap)
    dataset = EditDataset(graphs, nodes, edge_types)
    return dataset, amap if return_amap else amap


def load_data(data_dir):
    df_train = pandas.read_csv(
        os.path.join(data_dir, 'canonicalized_raw_train.csv')
    )

    print(df_train)


if __name__ == '__main__':
    load_data('../data/USPTO-50K')
