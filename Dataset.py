import torch
from typing import Any, Dict, List, Tuple, Optional, Union
import numpy as np
import torch_geometric
from numpy import concatenate as npcat


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
        self, graphs: List[Dict], activate_nodes: List[List[int]],
        changed_edges: List[List[Union[List[int], Tuple[int]]]],
        decoder_node_type: List[Dict], decoder_edge_type: List[Dict],
        rxn_class: Optional[List[int]] = None
    ):
        super(OverallDataset, self).__init__()
        self.graphs = graphs
        self.activate_nodes = activate_nodes
        self.changed_edges = changed_edges
        self.rxn_class = rxn_class
        self.decoder_node_class = decoder_node_type
        self.decoder_edge_class = decoder_edge_type
        # print(decoder_edge_type)
        # print(decoder_node_type)

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, index):
        this_graph = self.graphs[index]
        node_labels = torch.zeros(this_graph['num_nodes'])
        node_labels[self.activate_nodes[index]] = 1
        edges = this_graph['edge_index']
        edge_labels = torch.zeros(edges.shape[1])
        for idx, src in enumerate(edges[0]):
            src, dst = src.item(), edges[1][idx].item()
            if (src, dst) in self.changed_edges[index]:
                edge_labels[idx] = 1
            if (dst, src) in self.changed_edges[index]:
                edge_labels[idx] = 1

        if self.rxn_class is None:
            return this_graph, node_labels, edge_labels,\
                self.decoder_node_class[index], self.decoder_edge_class[index]

        else:
            return this_graph, node_labels, edge_labels,\
                self.rxn_class[index], self.decoder_node_class[index],\
                self.decoder_edge_class[index]


def make_decoder_graph(
    graphs, activate_nodes, changed_edges, pad_num, rxns=None,
    node_types=None, edge_types=None
):
    """[summary]

    a aux for decoder collate fn

    Args:
        graphs ([List]): a list of graphs, representing products
        activate_nodes ([List]): a list of tensors, each of tensor
            is of shape [num_nodes], representing whether a 
            node is activated on each graph
        changed_edges ([type]):  a list of tensors, eahc of tensor
            is of shape [num_edges], representing whether a edge is 
            changed on each graph
        pad_num ([int]): the number of pad nodes
        rxn ([list]): optional, a list of int representing the 
            reaction class (default: `None`)
        node_types ([list]): optional, a list of dict
            representing each atom type on reactant (default: `None`)
        edge_types ([list]): a list of dict, representing each
            bond type on reactant  (default: `None`)
    return:
        graphs (torch_geometric.data.Data): containing graph informations
        all_edge_types (dict): optional, a dict containing all the edge types, 
            edge indexes are merged
    """
    all_edge_types, all_node_types, org_edge = {}, [], []
    all_edg_idx, all_node_feat, all_edge_feat = [], [], []
    node_ptr, edge_ptr, node_batch, edge_batch = [0], [0], [], []
    node_rxn, edge_rxn, graph_rxn, lstnode, lstedge = [], [], [], 0, 0
    e_org_mask, e_pad_mask, attn_mask = [], [], []
    n_org_mask, n_pad_mask = [], []

    max_node = max(x['num_nodes'] + pad_num for x in graphs)
    batch_size = len(graphs)

    batch_mask = torch.zeros(batch_size, max_node).bool()

    for idx in range(batch_size):
        # init
        rxn = None if rxns is None else rxns[idx]
        o_n_cnt = graphs[idx]['num_nodes']
        a_n_cnt = o_n_cnt + pad_num

        batch_mask[idx, :a_n_cnt] = True

        # node

        all_node_feat.append(graphs[idx]['node_feat'])
        n_org_mask.append(torch.ones(o_n_cnt).bool())
        n_pad_mask.append(torch.zeros(o_n_cnt).bool())
        n_org_mask.append(torch.zeros(pad_num).bool())
        n_pad_mask.append(torch.ones(pad_num).bool())

        if node_types is not None:
            node_cls = torch.zeros(a_n_cnt).long()
            for k, v in node_types[idx].items():
                node_cls[k] = v
            all_node_types.append(node_cls)

        # org_edge

        edge_res_mask = (changed_edges[idx] == 0).tolist()
        res_enc_edges = graphs[idx]['edge_index'][:, edge_res_mask]
        o_e_cnt = res_enc_edges.shape[1]

        all_edge_feat.append(graphs[idx]['edge_feat'][edge_res_mask])
        all_edg_idx.append(res_enc_edges + lstnode)

        e_org_mask.append(torch.ones(o_e_cnt).bool())
        e_pad_mask.append(torch.zeros(o_e_cnt).bool())

        if edge_types is not None:
            org_edge_cls = torch.zeros(o_e_cnt).long()
            exists_edges = set()
            for edx in range(o_e_cnt):
                row, col = res_enc_edges[:, edx].tolist()
                exists_edges.add((row, col))
                org_edge_cls[edx] = edge_types[idx][(row, col)]
            org_edge.append(org_edge_cls)
        else:
            exists_edges = set()
            for edx in range(o_e_cnt):
                row, col = res_enc_edges[:, edx].tolist()
                exists_edges.add((row, col))

        # edge_labels

        if edge_types is not None:
            all_edge_types.update({
                (x + lstnode, y + lstnode): v for (x, y), v
                in edge_types[idx].items()
            })

        # pad_edges

        pad_idx = [x + o_n_cnt for x in range(pad_num)]

        attn_mask.append(make_attn_mask(res_enc_edges, max_node, pad_idx))

        prod_node_idx = torch.arange(0, o_n_cnt, 1)
        link_nds = prod_node_idx[activate_nodes[idx] == 1].tolist() + pad_idx

        pad_edges = [
            (x, y) for x in link_nds for y in link_nds
            if x != y and (x, y) not in exists_edges
        ]
        all_edg_idx.append(np.array(pad_edges, dtype=np.int64).T + lstnode)

        p_e_cnt = len(pad_edges)
        a_e_cnt = o_e_cnt + p_e_cnt

        e_org_mask.append(torch.zeros(p_e_cnt).bool())
        e_pad_mask.append(torch.ones(p_e_cnt).bool())

        lstnode += a_n_cnt
        lstedge += a_e_cnt

        node_batch.append(np.ones(a_n_cnt, dtype=np.int64) * idx)
        edge_batch.append(np.ones(a_e_cnt, dtype=np.int64) * idx)
        node_ptr.append(lstnode)
        edge_ptr.append(lstedge)

        if rxn is not None:
            graph_rxn.append(rxn)
            node_rxn.append(np.ones(a_n_cnt, dtype=np.int64) * rxn)
            edge_rxn.append(np.ones(a_e_cnt, dtype=np.int64) * rxn)

    result = {
        'x': torch.from_numpy(npcat(all_node_feat, axis=0)),
        'edge_attr': torch.from_numpy(npcat(all_edge_feat, axis=0)),
        'edge_index': torch.from_numpy(npcat(all_edg_idx, axis=1)),
        'attn_mask': torch.stack(attn_mask, dim=0),
        'batch': torch.from_numpy(npcat(node_batch, axis=0)),
        'e_batch': torch.from_numpy(npcat(edge_batch, axis=0)),
        "num_nodes": lstnode,
        "num_edges": lstedge,
        "ptr": torch.LongTensor(node_ptr),
        'e_ptr': torch.LongTensor(edge_ptr),
        'e_org_mask': torch.cat(e_org_mask, dim=0),
        'e_pad_mask': torch.cat(e_pad_mask, dim=0),
        'n_org_mask': torch.cat(n_org_mask, dim=0),
        "n_pad_mask": torch.cat(n_pad_mask, dim=0),
        'batch_mask': batch_mask
    }

    if len(graph_rxn) > 0:
        result['node_rxn'] = torch.from_numpy(npcat(node_rxn, axis=0))
        result['edge_rxn'] = torch.from_numpy(npcat(edge_rxn, axis=0))
        result['graph_rxn'] = torch.LongTensor(graph_rxn)

    if node_types is not None and edge_types is not None:
        result['node_class'] = torch.from_numpy(npcat(all_node_types, axis=0))
        result['org_edge_class'] = torch.from_numpy(npcat(org_edge, axis=0))
        return torch_geometric.data.Data(**result), all_edge_types
    else:
        return torch_geometric.data.Data(**result)


def make_attn_mask(edge_index, max_node, pad_idx):
    def dfs(x, graph, blocks, vis):
        blocks.append(x)
        vis.add(x)
        for neighbor in graph[x]:
            if neighbor not in vis:
                dfs(neighbor, graph, blocks, vis)

    def make_graph(edge_index, pad_idx, max_node):
        graph = {i: [i] for i in range(max_node) if i not in pad_idx}
        for idx in range(edge_index.shape[1]):
            row, col = edge_index[:, idx].tolist()
            if row not in graph:
                graph[row] = []
            graph[row].append(col)
        return graph

    attn_mask = torch.ones(max_node, max_node).bool()
    graph, vis = make_graph(edge_index, pad_idx, max_node), set()
    for node in graph.keys():
        if node not in vis:
            block = []
            dfs(node, graph, block, vis)
            block.extend(pad_idx)
            x_mask = torch.ones(max_node).bool()
            x_mask[block] = False
            block_attn = torch.zeros(max_node, max_node).bool()
            block_attn[x_mask] = True
            block_attn[:, x_mask] = True
            attn_mask &= block_attn
    return attn_mask


def overall_col_fn(pad_num):
    # use zero as empty type

    def col_fn(batch):
        use_class = len(batch[0]) == 6
        encoder_graph = edit_col_fn([x[:4] for x in batch])\
            if use_class else edit_col_fn([x[:3] for x in batch])

        # print('encoder done')
        paras = {
            'graphs': [x[0] for x in batch], 'pad_num': pad_num,
            'activate_nodes': [x[1] for x in batch],
            'changed_edges': [x[2] for x in batch],
            'node_types': [x[-2] for x in batch],
            'edge_types': [x[-1] for x in batch],
            'rxns': [x[3] for x in batch] if use_class else None
        }
        decoder_graph, all_edge_types = make_decoder_graph(**paras)

        return encoder_graph, decoder_graph, all_edge_types
    return col_fn


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
