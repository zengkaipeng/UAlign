import torch
from typing import Any, Dict, List, Tuple, Optional, Union
import numpy as np
import torch_geometric
from numpy import concatenate as npcat


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
            'x': torch.from_numpy(npcat(node_feat, axis=0)),
            "edge_attr": torch.from_numpy(npcat(edge_feat, axis=0)),
            'ptr': torch.LongTensor(node_ptr),
            'e_ptr': torch.LongTensor(edge_ptr),
            'batch': torch.from_numpy(npcat(node_batch, axis=0)),
            'e_batch': torch.from_numpy(npcat(edge_batch, axis=0)),
            'edge_index': torch.from_numpy(npcat(edge_idx, axis=-1)),
            "self_mask": torch.cat(self_mask, dim=0),
            'org_mask': torch.cat(org_mask, dim=0),
            'node_label': torch.cat(all_node, dim=0),
            'edge_label': torch.cat(all_edge, dim=0),
            'num_nodes': lstnode,
            'num_edges': lstedge,
            'attn_mask': torch.stack(attn_mask, dim=0)
        }

        for k, v in all_pos_enc.items():
            v = torch.from_numpy(npcat(v, axis=0))
            all_pos_enc[k] = v

        result.update(all_pos_enc)

        if len(node_rxn) > 0:
            node_rxn = npcat(node_rxn, axis=0)
            edge_rxn = npcat(edge_rxn, axis=0)
            result['node_rxn'] = torch.from_numpy(node_rxn)
            result['edge_rxn'] = torch.from_numpy(edge_rxn)

        return torch_geometric.data.Data(**result)

    return real_fn


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


def make_diag_by_mask(max_node, mask):
    x = torch.zeros(max_node, max_node)
    x[mask] = 1
    x[:, ~mask] = 0
    return x.bool()


def overall_col_fn(selfloop, pad_num):
    # use zero as empty type

    def dfs(x, graph, blocks, vis):
        blocks.append(x)
        vis.add(x)
        for neighbor in graph[x]:
            if neighbor not in vis:
                dfs(neighbor, graph, blocks, vis)

    def make_block(edge_index, max_node, pad_idx):
        print(max_node, pad_idx)
        attn_mask = torch.zeros(max_node, max_node).bool()
        graph, vis = {}, set()
        for idx in range(edge_index.shape[1]):
            row, col = edge_index[:, idx].tolist()
            if row not in graph:
                graph[row] = []
            graph[row].append(col)

        for node in graph.keys():
            if node in vis:
                continue
            block = []
            dfs(node, graph, block, vis)
            x_mask = torch.zeros(max_node).bool()
            print(block)
            x_mask[block] = True
            x_mask[pad_idx] = True
            block_attn = make_diag_by_mask(max_node, x_mask)
            attn_mask |= block_attn
        return attn_mask

    def col_fn(batch):
        encoder_fn = edit_col_fn(selfloop)
        use_class = len(batch[0]) == 6
        encoder_graph = encoder_fn([x[:4] for x in batch])\
            if use_class else encoder_fn([x[:3] for x in batch])

        # print('encoder done')

        all_edge_type, all_node, org_edge = {}, [], []
        all_edg_idx, all_node_feat, all_edge_feat = [], [], []
        node_ptr, edge_ptr, node_batch, edge_batch = [0], [0], [], []
        node_rxn, edge_rxn, graph_rxn, lstnode, lstedge = [], [], [], 0, 0
        self_mask, org_mask, pad_mask, attn_mask = [], [], [], []
        node_org_mask, node_pad_mask = [], []

        max_node = max(x[0]['num_nodes'] + pad_num for x in batch)
        # print('shape', [x[0]['num_nodes'] for x in batch])

        for idx, data in enumerate(batch):
            graph, n_lb, e_lb = data[:3]
            node_cls, edge_cls = data[-2:]
            node_cnt = graph['num_nodes'] + pad_num
            rxn = data[3] if use_class else None

            if rxn is not None:
                graph_rxn.append(rxn)

            # node_feats
            all_node_feat.append(graph['node_feat'])
            node_org_mask.append(torch.ones(graph['num_nodes']).bool())
            node_pad_mask.append(torch.zeros(graph['num_nodes']).bool())
            node_org_mask.append(torch.zeros(pad_num).bool())
            node_pad_mask.append(torch.ones(pad_num).bool())
            node_cls_x = np.zeros(node_cnt, dtype=np.int64)
            for k, v in node_cls.items():
                node_cls_x[k] = v
            all_node.append(node_cls_x)

            if rxn is not None:
                n_rxn = np.ones(graph['num_nodes'], dtype=np.int64) * rxn
                node_rxn.append(n_rxn)

            # org edge feat

            reserve_e_mask = (e_lb == 0).numpy()
            edge_cnt = int(reserve_e_mask.sum())
            all_edge_feat.append(graph['edge_feat'][reserve_e_mask])
            all_edg_idx.append(graph['edge_index'][:, reserve_e_mask])
            self_mask.append(torch.zeros(edge_cnt).bool())
            org_mask.append(torch.ones(edge_cnt).bool())
            pad_mask.append(torch.zeros(edge_cnt).bool())

            org_edge_cls = np.zeros(edge_cnt, dtype=np.int64)

            for idx in range(edge_cnt):
                row, col = all_edg_idx[-1][:, idx]
                org_edge_cls[idx] = edge_cls[(row, col)]

            org_edge.append(org_edge_cls)

            # self_loop edges
            if selfloop:
                all_edg_idx.append(torch.LongTensor([
                    list(range(node_cnt)), list(range(node_cnt))
                ]) + lstnode)
                edge_cnt += node_cnt
                self_mask.append(torch.ones(node_cnt).bool())
                org_mask.append(torch.zeros(node_cnt).bool())
                pad_mask.append(torch.zeros(node_cnt).bool())

            if rxn is not None:
                edge_rxn.append(np.ones(edge_cnt, dtype=np.int64) * rxn)

            # update_edge_types
            all_edge_type.update({
                (x + lstnode, y + lstnode): v
                for (x, y), v in edge_cls.items()
            })

            # make blocked attn block

            attn_mask.append(make_block(
                edge_index=graph['edge_index'][:, reserve_e_mask],
                max_node=max_node,
                pad_idx=[x + graph['num_nodes'] for x in range(pad_num)]
            ))

            # padded edge_idx

            prod_node_idx = np.arange(graph['num_nodes'], dtype=np.int64)
            link_nds = prod_node_idx[(n_lb == 1).numpy()].tolist()
            link_nds += [x + graph['num_nodes'] for x in range(pad_num)]
            pad_edges = [(x, y) for x in link_nds for y in link_nds if x != y]
            pad_len = len(pad_edges)
            self_mask.append(torch.zeros(pad_len).bool())
            org_mask.append(torch.zeros(pad_len).bool())
            pad_mask.append(torch.ones(pad_len).bool())
            edge_cnt += pad_len
            all_edg_idx.append(np.array(pad_edges, dtype=np.int64).T)

            lstnode += node_cnt
            lstedge += edge_cnt

            node_batch.append(np.ones(node_cnt, dtype=np.int64) * idx)
            edge_batch.append(np.ones(edge_cnt, dtype=np.int64) * idx)
            node_ptr.append(lstnode)
            edge_ptr.append(lstedge)

        result = {
            'x': torch.from_numpy(npcat(all_node_feat, axis=0)),
            'edge_attr': torch.from_numpy(npcat(all_edge_feat, axis=0)),
            'edge_index': torch.from_numpy(npcat(all_edg_idx, axis=1)),
            'attn_mask': torch.stack(attn_mask, dim=0),
            'node_class': torch.from_numpy(npcat(all_node, axis=0)),
            'batch': torch.from_numpy(npcat(node_batch, axis=0)),
            'e_batch': torch.from_numpy(npcat(edge_batch, axis=0)),
            "num_nodes": lstnode,
            "num_edges": lstedge,
            'org_edge_class': torch.from_numpy(npcat(org_edge, axis=0)),
            "ptr": torch.LongTensor(node_ptr),
            'e_ptr': torch.LongTensor(edge_ptr),
            "self_mask": torch.cat(self_mask, dim=0),
            'org_mask': torch.cat(org_mask, dim=0),
            'pad_mask': torch.cat(pad_mask, dim=0),
            'node_org_mask': torch.cat(node_org_mask, dim=0),
            "node_pad_mask": torch.cat(node_pad_mask, dim=0)
        }
        if len(graph_rxn) > 0:
            result['node_rxn'] = torch.from_numpy(npcat(node_rxn, axis=0))
            result['edge_rxn'] = torch.from_numpy(npcat(edge_rxn, axis=0))
            result['graph_rxn'] = torch.LongTensor(graph_rxn)

        decoder_graph = torch_geometric.data.Data(**result)

        return encoder_graph, decoder_graph, all_edge_type
    return col_fn
