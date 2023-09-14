import torch
from typing import Any, Dict, List, Tuple, Optional, Union
import numpy as np
import torch_geometric


class BinaryEditDataset(torch.utils.data.Dataset):
    def __init__(
        self, graphs: List[Dict], activate_nodes: List[List[int]],
        changed_edges: List[List[Union[List[int], Tuple[int]]]],
        rxn_class: Optional[List[int]] = None
    ):
        super(BinaryEditDataset, self).__init__()
        self.graphs = graphs
        self.activate_nodes = activate_nodes
        self.changed_edges = changed_edges
        self.rxn_class = rxn_class

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, index):
        node_labels = torch.zeros(self.graphs[index]['num_nodes'])
        node_labels[self.activate_nodes[index]] = 1
        edges = self.graphs[index]['edge_index']
        edge_labels = torch.zeros(edges.shape[1])
        for idx, src in enumerate(edges[0]):
            src, dst = src.item(), edges[1][idx].item()
            if (src, dst) in self.changed_edges[index]:
                edge_labels[idx] = 1
            if (dst, src) in self.changed_edges[index]:
                edge_labels[idx] = 1

        if self.rxn_class is None:
            return self.graphs[index], node_labels, edge_labels
        else:
            return self.graphs[index], node_labels, \
                edge_labels, self.rxn_class[index]


def edit_col_fn(selfloop):
    def add_list_to_dict(k, v, it):
        if k not in it:
            it[k] = [v]
        else:
            it[k].append(v)

    def real_fn(batch):
        batch_size, all_node, all_edge = len(batch), [], []
        edge_idx, node_feat, edge_feat = [], [], []
        node_ptr, edge_ptr, node_batch, edge_batch = [0], [0], [], []
        node_rxn, edge_rxn, lstnode, lstedge = [], [], 0, 0
        self_mask, org_mask, attn_mask = [], [], []

        all_pos_enc = {}
        max_node = max(x[0]['num_nodes'] for x in batch)

        for idx, data in enumerate(batch):
            if len(data) == 4:
                gp, nlb, elb, rxn = data
            else:
                (gp, nlb, elb), rxn = data, None

            node_cnt, edge_cnt = gp['num_nodes'], gp['edge_index'].shape[1]

            for k, v in gp.items():
                if 'pos_enc' in k:
                    add_list_to_dict(k, v, all_pos_enc)

            node_feat.append(gp['node_feat'])
            edge_feat.append(gp['edge_feat'])
            edge_idx.append(gp['edge_index'] + lstnode)
            self_mask.append(torch.zeros(edge_cnt).bool())
            org_mask.append(torch.ones(edge_cnt).bool())
            all_node.append(nlb)
            all_edge.append(elb)

            ams = torch.zeros((max_node, max_node))
            ams[:node_cnt, :node_cnt] = 1
            attn_mask.append(ams.bool())

            if selfloop:
                edge_idx.append(torch.LongTensor([
                    list(range(node_cnt)), list(range(node_cnt))
                ]) + lstnode)
                edge_cnt += node_cnt
                self_mask.append(torch.ones(node_cnt).bool())
                org_mask.append(torch.zeros(node_cnt).bool())
                all_edge.append(torch.zeros(node_cnt))

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
            'x': torch.from_numpy(np.concatenate(node_feat, axis=0)),
            "edge_attr": torch.from_numpy(np.concatenate(edge_feat, axis=0)),
            'ptr': torch.LongTensor(node_ptr),
            'e_ptr': torch.LongTensor(edge_ptr),
            'batch': torch.from_numpy(np.concatenate(node_batch, axis=0)),
            'e_batch': torch.from_numpy(np.concatenate(edge_batch, axis=0)),
            'edge_index': torch.from_numpy(np.concatenate(edge_idx, axis=-1)),
            "self_mask": torch.cat(self_mask, dim=0),
            'org_mask': torch.cat(org_mask, dim=0),
            'node_label': torch.cat(all_node, dim=0),
            'edge_label': torch.cat(all_edge, dim=0),
            'num_nodes': lstnode,
            'num_edges': lstedge,
            'attn_mask': torch.stack(attn_mask, dim=0)
        }

        for k, v in all_pos_enc.items():
            v = torch.from_numpy(np.concatenate(v, axis=0))
            all_pos_enc[k] = v

        result.update(all_pos_enc)

        if len(node_rxn) > 0:
            node_rxn = np.concatenate(node_rxn, axis=0)
            edge_rxn = np.concatenate(edge_rxn, axis=0)
            result['node_rxn'] = torch.from_numpy(node_rxn)
            result['edge_rxn'] = torch.from_numpy(edge_rxn)

        return torch_geometric.data.Data(**result)

    return real_fn


class OverallDataset(torch.utils.data.Dataset):
    def __init__(
        self, graphs, activate_nodes, changed_edges,
        decoder_node_type, decoder_edge_type, rxn_class=None
    ):
        super(OverallDataset, self).__init__()
        self.graphs = graphs
        self.activate_nodes = activate_nodes
        self.changed_edges = changed_edges
        self.rxn_class = rxn_class
        self.pad_num = pad_num
        self.decoder_node_class = decoder_node_type
        self.decoder_edge_class = decoder_edge_type

    def __init__(self):
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


def overall_col_fn(selfloop, pad_num):
    # use zero as empty type
    def col_fn(batch):
        encoder_fn = edit_col_fn(selfloop)
        use_class = len(batch[0]) == 6
        encoder_graph = encoder_fn([x[:3] for x in batch])\
            if use_class else encoder_fn([x[:2] for x in batch])

        all_edge_type, all_node = {}, []
        all_edg_idx, all_node_feat, all_edge_feat = [], [], []
        node_ptr, edge_ptr, node_batch, edge_batch = [0], [0], [], []
        node_rxn, edge_rxn, graph_rxn, lstnode, lstedge = [], [], [], 0, 0

        for idx, data in enumerate(batch):
            graph, n_lb, e_lb = data[:3]
            prod_node_idx = np.arange(graph['num_nodes'])
            link_nodes = prod_node_idx[(n_lb == 1).numpy()]
            link_nodes += [x + graph['num_nodes'] for x in range(pad_num)]

            
