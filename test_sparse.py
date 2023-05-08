from backBone import  EditDataset
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

Loader = DataLoader(dataset, collate_fn=sparse_edit_collect_fn, batch_size=4)

GIN = GINBase(num_layers=4, embedding_dim=256, dropout=0.1, edge_last=False)
GAT = GATBase(num_heads=4, num_layers=4, embedding_dim=256, edge_last=False)

model1 = GraphEditModel(GIN, True, 256, 256, 4)

model2 = GraphEditModel(GAT, True, 256, 256, 4)

for x in Loader:
    graphs, node_label, num_l, num_e, e_type, act_nodes = x
    print('[DATA]', x)
    node_res, edge_res, new_act = model1(
        graphs=graphs, act_nodes=act_nodes, num_nodes=num_l, num_edges=num_e,
        mode='together'
    ) 
    print(node_res.shape, edge_res.shape, new_act)