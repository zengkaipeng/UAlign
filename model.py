import torch
from sparse_backBone import (
    GINBase, GATBase, SparseAtomEncoder, SparseBondEncoder
)

from typing import Any, Dict, List, Tuple, Optional, Union
from torch_geometric.data import Data as GData


class EditDataset(torch.utils.data.Dataset):
    def __init__(
        self, graphs: List[Dict],
        activate_nodes: List[List],
        activate_edges: List[List],
        reat: List[str],
        rxn_class: Optional[List[int]] = None
    ):
        super(EditDataset, self).__init__()
        self.graphs = graphs
        self.activate_nodes = activate_nodes
        self.activate_edges = activate_edges
        self.rxn_class = rxn_class
        self.reat = reat

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

        if self.rxn_class is None:
            ret = ['<CLS>']
        else:
            ret = [f'<RXN_{self.rxn_class[index]}>']
        ret += list(self.reat[index])

        if self.rxn_class is not None:
            return self.graphs[index], self.rxn_class[index], \
                node_label, edge_label, ret
        else:
            return self.graphs[index], node_label, edge_label, ret


def collect_fn(data_batch):
    batch_size, rxn_class, node_label = len(data_batch), [], []
    edge_idxes, edge_feats, node_feats, lstnode = [], [], [], 0
    edge_label, batch, ptr = [], [], [0]
    edge_rxn, node_rxn, reats = [], [], []
    for idx, data in enumerate(data_batch):
        if len(data) == 4:
            graph, n_lb, e_lb, ret = data
        else:
            graph, r_class, n_lb, e_lb, ret = data
            rxn_class.append(r_class)
            node_rxn.append(
                np.ones(graph['num_nodes'], dtype=np.int64) * r_class
            )
            edge_rxn.append(
                np.ones(graph['edge_index'].shape[1], dtype=np.int64) * r_class
            )

        node_label.append(n_lb)
        edge_label.append(e_lb)
        reats.append(ret)

        edge_idxes.append(graph['edge_index'] + lstnode)
        edge_feats.append(graph['edge_feat'])
        node_feats.append(graph['node_feat'])
        lstnode += graph['num_nodes']
        batch.append(np.ones(graph['num_nodes'], dtype=np.int64) * idx)
        ptr.append(lstnode)

    result = {
        'edge_index': np.concatenate(edge_idxes, axis=-1),
        'edge_attr': np.concatenate(edge_feats, axis=0),
        'batch': np.concatenate(batch, axis=0),
        'x': np.concatenate(node_feats, axis=0)
        'ptr': np.array(ptr, dtype=np.int64)
    }

    if len(rxn_class) != 0:
        result['node_rxn'] = np.concatenate(node_rxn, axis=0)
        result['edge_rxn'] = np.concatenate(edge_rxn, axis=0)
        result['rxn_class'] = np.array(rxn_class, dtype=np.int64)

    result = {k: torch.from_numpy(v) for k, v in result.items()}
    result['num_nodes'] = lstnode
    node_label = torch.cat(node_label, dim=0)
    edge_label = torch.cat(edge_label, dim=0)

    return Data(**result), node_label, edge_label, reats
