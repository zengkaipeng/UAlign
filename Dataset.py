import torch
from typing import Any, Dict, List, Tuple, Optional, Union
import numpy as np
import torch_geometric
from numpy import concatenate as npcat
from tokenlizer import smi_tokenizer


class SynthonDataset(torch.utils.data.Dataset):
    def __init__(
        self, graphs: List[Dict], nodes_label: List[Dict],
        edges_label: List[Dict[Tuple[int, int], int]],
        rxn_class: Optional[List[int]] = None
    ):
        super(SynthonDataset, self).__init__()
        self.graphs = graphs
        self.node_labels = nodes_label
        self.edge_labels = edges_label
        self.rxn_class = rxn_class

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, index):
        num_nodes = self.graphs[index]['node_feat'].shape[0]
        num_edges = self.graphs[index]['edge_index'].shape[1]
        node_labels = torch.zeros(num_nodes).long()
        edge_labels = torch.zeros(num_edges).long()

        for k, v in self.node_labels[index].items():
            node_labels[k] = v
        for idx in range(num_edges):
            row, col = self.graphs[index]['edge_index'][:, idx].tolist()
            edge_labels[idx] = self.edge_labels[index][(row, col)]

        if self.rxn_class is None:
            return self.graphs[index], node_labels, edge_labels
        else:
            return self.graphs[index], node_labels, edge_labels, \
                self.rxn_class[index]


def edit_col_fn(batch):
    batch_size, all_node, all_edge = len(batch), [], []
    edge_idx, node_feat, edge_feat = [], [], []
    node_ptr, edge_ptr, node_batch, edge_batch = [0], [0], [], []
    node_rxn, edge_rxn, lstnode, lstedge = [], [], 0, 0
    max_node = max(x[0]['num_nodes'] for x in batch)
    batch_mask = torch.zeros(batch_size, max_node).bool()

    for idx, data in enumerate(batch):
        if len(data) == 4:
            gp, nlb, elb, rxn = data
        else:
            (gp, nlb, elb), rxn = data, None

        node_cnt, edge_cnt = gp['num_nodes'], gp['edge_index'].shape[1]

        node_feat.append(gp['node_feat'])
        edge_feat.append(gp['edge_feat'])
        edge_idx.append(gp['edge_index'] + lstnode)
        all_node.append(nlb)
        all_edge.append(elb)

        batch_mask[idx, :node_cnt] = True

        lstnode += node_cnt
        lstedge += edge_cnt
        node_batch.append(np.ones(node_cnt, dtype=np.int64) * idx)
        edge_batch.append(np.ones(edge_cnt, dtype=np.int64) * idx)
        node_ptr.append(lstnode)
        edge_ptr.append(lstedge)

        if rxn is not None:
            node_rxn.append(np.ones(node_cnt, dtype=np.int64) * rxn)
            edge_rxn.append(np.ones(edge_cnt, dtype=np.int64) * rxn)

    result = {
        'x': torch.from_numpy(npcat(node_feat, axis=0)),
        "edge_attr": torch.from_numpy(npcat(edge_feat, axis=0)),
        'ptr': torch.LongTensor(node_ptr),
        'e_ptr': torch.LongTensor(edge_ptr),
        'batch': torch.from_numpy(npcat(node_batch, axis=0)),
        'e_batch': torch.from_numpy(npcat(edge_batch, axis=0)),
        'edge_index': torch.from_numpy(npcat(edge_idx, axis=-1)),
        'node_label': torch.cat(all_node, dim=0),
        'edge_label': torch.cat(all_edge, dim=0),
        'num_nodes': lstnode,
        'num_edges': lstedge,
        'batch_mask': batch_mask
    }

    if len(node_rxn) > 0:
        node_rxn = npcat(node_rxn, axis=0)
        edge_rxn = npcat(edge_rxn, axis=0)
        result['node_rxn'] = torch.from_numpy(node_rxn)
        result['edge_rxn'] = torch.from_numpy(edge_rxn)

    return torch_geometric.data.Data(**result)


class OverallDataset(torch.utils.data.Dataset):
    def __init__(
        self, graphs: List[Dict], enc_nodes: List[List[int]],
        enc_edges: List[Dict[Tuple[int, int], int]],
        lg_graphs: List[Dict], lg_labels: List[List[int]],
        conn_edges: List[List[List[int]]], conn_labels: List[List[int]],
        trans_input: List[str], trans_output: List[str],
        rxn_class: Optional[List[int]] = None
    ):
        super(OverallDataset, self).__init__()
        self.graphs = graphs
        self.node_labels = nodes_label
        self.edge_labels = edges_label
        self.rxn_class = rxn_class
        self.lg_graphs = lg_graphs
        self.lg_labels = lg_labels
        self.conn_edges = conn_edges
        self.conn_labels = conn_labels
        self.trans_input = trans_input
        self.trans_output = trans_output

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, index):
        num_nodes = self.graphs[index]['node_feat'].shape[0]
        num_edges = self.graphs[index]['edge_index'].shape[1]
        node_labels = torch.zeros(num_nodes).long()
        edge_labels = torch.zeros(num_edges).long()

        for k, v in self.node_labels[index].items():
            node_labels[k] = v
        for idx in range(num_edges):
            row, col = self.graphs[index]['edge_index'][:, idx].tolist()
            edge_labels[idx] = self.edge_labels[index][(row, col)]

        if self.rxn_class is not None:
            trans_output = [f'<RXN_{rxn_class}>']
        else:
            trans_output = [f'<CLS>']

        trans_input = smi_tokenizer(self.trans_input[index])
        trans_output.extend(smi_tokenizer(self.trans_output[index]))
        trans_output.append('<END>')

        if self.rxn_class is None:
            return self.graphs[index], node_labels, edge_labels, \
                self.lg_graphs[index], self.lg_labels[index],\
                self.conn_edges[index], self.conn_labels[index],\
                trans_input, trans_output
        else:
            return self.graphs[index], node_labels, edge_labels, \
                self.lg_graphs[index], self.lg_labels[index],\
                self.conn_edges[index], self.conn_labels[index],\
                trans_input, trans_output, self.rxn_class[index]


def overall_col_fn(batch):
    encoder = {
        'node_feat': [], 'edge_feat': [], 'node_label': [],
        'edge_label': [], 'edge_index': [], 'node_batch': [],
        'edge_batch': [], 'lstnode': 0, 'lstedge': 0,
        'node_ptr': [0], 'edge_ptr': [0], 'node_rxn': [],
        'edge_rxn': []
    }

    LG = {
        'node_feat': [], 'edge_feat': [], 'node_label': [],
        'edge_label': [], 'edge_index': [], 'node_batch': [],
        'edge_batch': [], 'lstnode': 0, 'lstedge': 0,
        'node_ptr': [0], 'edge_ptr': [0], 'node_rxn': [],
        'edge_rxn': []
    }

    batch_size, graph_rxn = len(batch), []
    conn_edges, conn_labels = [], []
    trans_input, trans_output = [], []

    encoder['max_node'] = max(x[0]['num_nodes'] for x in batch)
    encoder['batch_mask'] = torch.zeros(batch_size, encoder['max_node']).bool()

    LG['max_node'] = max(x[4]['num_nodes'] for x in batch)
    LG['batch_mask'] = torch.zeros(batch_size, LG['max_node']).bool()

    for idx, data in enumerate(batch):
        if len(data) == 10:
            gp, nlb, elb, lgg, lgb, coe, col, tin, tou, rxn = data
        else:
            (gp, nlb, elb, lgg, lgb, coe, col, tin, tou, rxn), rxn = data, None

        # encoder
        enc_node_cnt = gp['num_nodes']
        enc_edge_cnt = gp['edge_index'].shape[1]
        encoder['node_feat'].append(gp['node_feat'])
        encoder['edge_feat'].append(gp['edge_feat'])
        encoder['edge_index'].append(gp['edge_index'] + encoder['lstnode'])
        encoder['node_label'].append(nlb)
        encoder['edge_label'].append(elb)
        encoder['batch_mask'][idx, :enc_node_cnt] = True
        encoder['node_batch'].append(
            np.ones(enc_node_cnt, dtype=np.int64) * idx
        )
        encoder['edge_batch'].append(
            np.ones(enc_edge_cnt, dtype=np.int64) * idx
        )

        # lg

        lg_node_cnt = lgg['num_nodes']
        lg_edge_cnt = lgg['edge_index'].shape[1]
        LG['node_feat'].append(lgg['node_feat'])
        LG['edge_feat'].append(lgg['edge_feat'])
        LG['edge_index'].append(lgg['edge_index'] + LG['lstnode'])
        LG['node_label'].extend(lgb)
        LG['batch_mask'][idx, :lg_edge_cnt] = True
        LG['node_batch'].append(np.ones(lg_node_cnt, dtype=np.int64) * idx)
        LG['edge_batch'].append(np.ones(lg_edge_cnt, dtype=np.int64) * idx)

        # conn
        conn_edges.extend([
            (a + encoder['lstnode'], b + LG['lstnode']) for a, b in coe
        ])
        conn_labels.extend(col)

        # trans
        trans_input.append(tin)
        trans_output.append(tout)

        # lst update

        encoder['lstnode'] += enc_node_cnt
        encoder['lstedge'] += enc_edge_cnt
        encoder['node_batch'].append(
            np.ones(enc_node_cnt, dtype=np.int64) * idx
        )
        encoder['edge_batch'].append(
            np.ones(enc_edge_cnt, dtype=np.int64) * idx
        )
        encoder['node_ptr'].append(encoder['lstnode'])
        encoder['edge_ptr'].append(encoder['lstedge'])

        LG['lstnode'] += lg_node_cnt
        LG['lstedge'] += lg_edge_cnt
        LG['node_batch'].append(np.ones(lg_node_cnt, dtype=np.int64) * idx)
        LG['edge_batch'].append(np.ones(lg_edge_cnt, dtype=np.int64) * idx)
        LG['node_ptr'].append(LG['lstnode'])
        LG['edge_ptr'].append(LG['lstedge'])

        if rxn is not None:
            encoder['node_rxn'].append(
                np.ones(enc_node_cnt, dtype=np.int64) * rxn
            )
            encoder['edge_rxn'].append(
                np.ones(enc_edge_cnt, dtype=np.int64) * rxn
            )
            LG['node_rxn'].append(np.ones(lg_node_cnt, dtype=np.int64) * rxn)
            LG['edge_rxn'].append(np.ones(lg_edge_cnt, dtype=np.int64) * rxn)
            graph_rxn.append(rxn)

    enc_graph = {
        'x': torch.from_numpy(npcat(encoder['node_feat'], axis=0)),
        'edge_attr': torch.from_numpy(npcat(encoder['edge_feat'], axis=0)),
        'ptr': torch.LongTensor(encoder['node_ptr']),
        'e_ptr': torch.LongTensor(encoder['edge_ptr']),
        'batch': torch.from_numpy(npcat(encoder['node_batch'], axis=0)),
        'e_batch': torch.from_numpy(npcat(encoder['edge_batch'], axis=0)),
        'edge_index': torch.from_numpy(npcat(encoder['edge_index'], axis=1)),
        'node_label': torch.cat(encoder['node_label'], dim=0),
        'edge_label': torch.cat(encoder['edge_label'], dim=0),
        'num_nodes': encoder['lstnode'],
        'num_edges': encoder['lstedge'],
        "batch_mask": encoder['batch_mask']
    }

    lg_graph = {
        'x': torch.from_numpy(npcat(LG['node_feat'], axis=0)),
        'edge_attr': torch.from_numpy(npcat(LG['edge_feat'], axis=0)),
        'ptr': torch.LongTensor(LG['node_ptr']),
        'e_ptr': torch.LongTensor(LG['edge_ptr']),
        'batch': torch.from_numpy(npcat(LG['node_batch'], axis=0)),
        'e_batch': torch.from_numpy(npcat(LG['edge_batch'], axis=0)),
        'edge_index': torch.from_numpy(npcat(LG['edge_index'], axis=1)),
        'node_label': torch.FloatTensor(LG['node_label']),
        'num_nodes': LG['lstnode'],
        'num_edges': LG['lstedge'],
        "batch_mask": LG['batch_mask']
    }

    if len(graph_rxn) > 0:
        enc_graph['node_rxn'] = torch.from_numpy(npcat(
            encoder['node_rxn'], axis=0
        ))
        enc_graph['edge_rxn'] = torch.from_numpy(npcat(
            encoder['edge_rxn'], axis=0
        ))
        lg_graph['node_rxn'] = torch.from_numpy(npcat(LG['node_rxn'], axis=0))
        lg_graph['edge_rxn'] = torch.from_numpy(npcat(LG['edge_rxn'], axis=0))
        graph_rxn = torch.LongTensor(graph_rxn)
    else:
        graph_rxn = None

    conn_edges = torch.LongTensor(conn_edges)
    conn_labels = torch.FloatTensor(conn_labels)

    return torch_geometric.data.Data(**enc_graph), \
        torch_geometric.data.Data(**lg_graph),\
        conn_edges, conn_labels, trans_input, trans_output, graph_rxn


class InferenceDataset(torch.utils.data.Dataset):
    def __init__(
        self, reac_graph, prod_smiles, reac_node_type,
        reac_edge_type, rxn_class=None
    ):
        super(InferenceDataset, self).__init__()

        self.reac_graph = reac_graph
        self.prod_smiles = prod_smiles
        self.reac_node_type = reac_node_type
        self.reac_edge_type = reac_edge_type
        self.rxn_class = rxn_class

    def __len__(self):
        return len(self.reac_graph)

    def __getitem__(self, index):
        answer = (
            self.reac_graph[index], self.prod_smiles[index],
            self.reac_node_type[index], self.reac_edge_type[index]
        )

        if self.rxn_class is not None:
            answer += (self.rxn_class[index], )
        return answer


def inference_col_fn(batch):
    batch_size = len(batch)
    edge_idx, node_feat, edge_feat = [], [], []
    node_ptr, edge_ptr, node_batch, edge_batch = [0], [0], [], []
    node_rxn, edge_rxn, lstnode, lstedge = [], [], 0, 0
    node_types, edge_types, smiles = [], [], []

    max_node = max(x[0]['num_nodes'] for x in batch)
    batch_mask = torch.zeros(batch_size, max_node).bool()

    for idx, data in enumerate(batch):
        if len(data) == 4:
            (gp, smi, n_type, e_type), rxn = data, None
        else:
            gp, smi, n_type, e_type, rxn = data

        node_types.append(n_type)
        edge_types.append(e_type)
        smiles.append(smi)

        node_cnt, edge_cnt = gp['num_nodes'], gp['edge_index'].shape[1]

        batch_mask[idx, : node_cnt] = True

        node_feat.append(gp['node_feat'])
        edge_feat.append(gp['edge_feat'])
        edge_idx.append(gp['edge_index'] + lstnode)

        lstnode += node_cnt
        lstedge += edge_cnt
        node_batch.append(np.ones(node_cnt, dtype=np.int64) * idx)
        edge_batch.append(np.ones(edge_cnt, dtype=np.int64) * idx)
        node_ptr.append(lstnode)
        edge_ptr.append(lstedge)

        if rxn is not None:
            node_rxn.append(np.ones(node_cnt, dtype=np.int64) * rxn)
            edge_rxn.append(np.ones(edge_cnt, dtype=np.int64) * rxn)

    result = {
        'x': torch.from_numpy(npcat(node_feat, axis=0)),
        "edge_attr": torch.from_numpy(npcat(edge_feat, axis=0)),
        'ptr': torch.LongTensor(node_ptr),
        'e_ptr': torch.LongTensor(edge_ptr),
        'batch': torch.from_numpy(npcat(node_batch, axis=0)),
        'e_batch': torch.from_numpy(npcat(edge_batch, axis=0)),
        'edge_index': torch.from_numpy(npcat(edge_idx, axis=-1)),
        'num_nodes': lstnode,
        'num_edges': lstedge,
        'batch_mask': batch_mask
    }

    if len(node_rxn) > 0:
        node_rxn = npcat(node_rxn, axis=0)
        edge_rxn = npcat(edge_rxn, axis=0)
        result['node_rxn'] = torch.from_numpy(node_rxn)
        result['edge_rxn'] = torch.from_numpy(edge_rxn)

    return torch_geometric.data.Data(**result), \
        node_types, edge_types, smiles
