from model import EditDataset, get_collate_fn
from sparse_backBone import sparse_edit_collect_fn
from utils.chemistry_parse import (
    get_reaction_core, get_bond_info, BOND_FLOAT_TO_TYPE,
    BOND_FLOAT_TO_IDX
)
from utils.graph_utils import smiles2graph
from torch.utils.data import DataLoader
from model import GraphEditModel
from sparse_backBone import GINBase, GATBase


with open('utils/test_examples.txt') as Fin:
    content = Fin.readlines()


graphs, nodes, edge = [], [], []

for react in content:
    if len(react) <= 1:
        continue
    reac, prod = react.strip().split('>>')
    x, y, z = get_reaction_core(reac, prod)
    graph, amap = smiles2graph(prod, with_amap=True)
    graphs.append(graph)
    nodes.append([amap[t] for t in x])
    edge_types = {
        (amap[i], amap[j]): BOND_FLOAT_TO_IDX[v[0]]
        for (i, j), v in z.items() if i in amap and j in amap
    }
    edge.append(edge_types)


dataset = EditDataset(graphs, nodes, edge)

cfn = get_collate_fn(sparse=True, self_loop=True)

Loader = DataLoader(dataset, collate_fn=cfn, batch_size=4)

GIN = GINBase(num_layers=4, embedding_dim=256, dropout=0.1, edge_last=False)

model1 = GraphEditModel(GIN, True, 256, 256, 4)

for x in Loader:
    graphs, node_label, e_type, act_nodes, e_map = x
    print('[DATA]', x)
    print('[node_label_shape]', node_label.shape)
