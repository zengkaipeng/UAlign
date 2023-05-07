from model import edit_collect_fn, EditDataset
from utils.chemistry_parse import (
    get_reaction_core, get_bond_info, BOND_FLOAT_TO_TYPE,
    BOND_FLOAT_TO_IDX
)
from utils.graph_utils import smiles2graph
from torch.utils.data import DataLoader


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

Loader = DataLoader(dataset, collate_fn=edit_collect_fn, batch_size=2)
for x in Loader:
    print(x)
