import torch
from typing import Any, Dict, List, Tuple, Optional, Union
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from torch_geometric.data import Data


class EditDataset(torch.utils.data.Dataset):
    def __init__(
        self, graphs: List[Dict],
        activate_nodes: List[List],
        activate_edges: List[List],
        rxn_class: Optional[List[int]] = None
    ):
        super(EditDataset, self).__init__()
        self.graphs = graphs
        self.activate_nodes = activate_nodes
        self.activate_edges = activate_edges
        self.rxn_class = rxn_class

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, index):
        node_label = torch.zeros(self.graphs[index]['num_nodes']).long()
        node_label[self.activate_nodes[index]] = 1
        num_edges = self.graphs[index]['edge_attr'].shape[0]
        edge_label = torch.zeros(num_edges).long()
        edges = self.graphs[index]['edge_index']
        for idx, t in enumerate(edges[0]):
            src, dst = t.item(), edges[1][idx].item()
            if (src, dst) in self.activate_edges[index]:
                edge_label[idx] = 1
            if (dst, src) in self.activate_edges[index]:
                edge_label[idx] = 1

        if self.rxn_class is not None:
            return self.graphs[index], self.rxn_class[index], \
                node_label, edge_label
        else:
            return self.graphs[index], node_label, edge_label


def edit_collect_fn(data_batch):
    batch_size = len(data_batch)
    max_node = max([x[0]['num_nodes'] for x in data_batch])
    attn_mask = torch.zeros(batch_size, max_node, max_node, dtype=bool)
    node_label = torch.ones(batch_size, max_node) * -100
    edge_cores, edge_types, graphs, rxn_class = [], [], [], []
    for idx, data in enumerate(data_batch):
        if len(data) == 4:
            graph, n_lb, e_type, e_core = data
        else:
            graph, r_class, n_lb, e_type, e_core = data
            rxn_class.append(r_class)
        node_num = graph['num_nodes']
        node_label[idx][:node_num] = n_lb
        attn_mask[idx][:node_num, :node_num] = True
        edge_cores.append(e_core)
        edge_types.append(e_type)

        graph['node_feat'] = torch.from_numpy(graph['node_feat']).float()
        graph['edge_feat'] = torch.from_numpy(graph['edge_feat']).float()
        graph['edge_index'] = torch.from_numpy(graph['edge_index'])
        graphs.append(Data(**graph))

    if len(rxn_class) == 0:
        return attn_mask, graphs, node_label, edge_cores, edge_types
    else:
        return attn_mask, graphs, torch.LongTensor(rxn_class),\
            node_label, edge_cores, edge_types
