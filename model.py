import torch
from backBone import FCGATEncoder, ExtendedAtomEncoder, ExtendedBondEncoder
from sparse_backBone import (
    GINBase, GATBase, SparseAtomEncoder, SparseBondEncoder
)
from itertools import combinations, permutations
from torch_geometric.data import Data
from typing import Any, Dict, List, Tuple, Optional, Union
import numpy as np


class GraphEditModel(torch.nn.Module):
    def __init__(
        self, base_model, is_sparse, node_dim, edge_dim,
        edge_class,  dropout=0.1
    ):
        super(GraphEditModel, self).__init__()
        self.base_model = base_model
        self.sparse = is_sparse

        if self.sparse:
            self.edge_feat_agger = torch.nn.Linear(
                node_dim + node_dim, edge_dim
            )
        self.edge_predictor = torch.nn.Sequential(
            torch.nn.Linear(edge_dim, edge_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(edge_dim, edge_class)
        )

        self.node_predictor = torch.nn.Sequential(
            torch.nn.Linear(node_dim, node_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(node_dim, 2)
        )
        if self.sparse:
            self.atom_encoder = SparseAtomEncoder(node_dim)
            self.bond_encoder = SparseBondEncoder(edge_dim)
        else:
            self.atom_encoder = ExtendedAtomEncoder(node_dim)
            self.edge_encoder = ExtendedBondEncoder(edge_dim)

    def get_edge_feat(self, node_feat, edge_index):
        assert self.sparse, 'Only sparse mode have edge_feat_agger'
        src_x = torch.index_select(node_feat, dim=0, index=edge_index[:, 0])
        dst_x = torch.index_select(node_feat, dim=0, index=edge_index[:, 1])
        return self.edge_feat_agger(torch.cat([src_x, dst_x], dim=-1))

    def predict_edge(
        self, node_feat, activate_nodes, edge_feat=None,
        e_types=None, edge_map=None, empty_type=0, ptr=None
    ):
        if self.sparse and ptr is None:
            raise NotImplementedError(
                'sparse backbone requires number of each graph '
                'to obtain correct edge features'
            )
        if not self.sparse and (edge_feat is None or edge_map is None):
            raise NotImplementedError(
                'dense backbone calculated every pair of edge_features '
                'and the edge features should be provided'
            )
        e_answer, e_ptr = [], [0]

        if self.sparse:
            src_idx, dst_idx = [], []
            for idx, p in enumerate(activate_nodes):
                # print('[Anodes]', p)
                for x, y in combinations(p, 2):
                    src_idx.append((x + ptr[idx], y + ptr[idx]))
                    dst_idx.append((y + ptr[idx], x + ptr[idx]))
                    e_answer.append(get_label(e_types[idx], x, y, empty_type))
                e_ptr.append(len(e_answer))

            if len(src_idx) == 0:
                return None, [], [0]

            src_idx = torch.LongTensor(src_idx).to(node_feat.device)
            dst_idx = torch.LongTensor(dst_idx).to(node_feat.device)

            ed_feat = self.get_edge_feat(node_feat, src_idx) + \
                self.get_edge_feat(node_feat, dst_idx)

        else:
            src_idx, dst_idx = [], []
            for idx, p in enumerate(activate_nodes):
                for x, y in combinations(p, 2):
                    src_idx.append(edge_map[(x, y)])
                    dst_idx.append(edge_map[(y, x)])
                    e_answer.append(get_label(e_types[idx], x, y, empty_type))
                e_ptr.append(len(e_answer))

            if len(src_idx) == 0:
                return None, [], [0]

            ed_feat = edge_feat[src_idx] + edge_feat[dst_idx]

        return self.edge_predictor(ed_feat), e_answer, e_ptr

    def get_init_feats(self, graphs):
        rxn_node = getattr(graphs, 'rxn_node', None)
        rxn_edge = getattr(graphs, 'rxn_edge', None)
        node_feat = self.atom_encoder(node_feat=graphs.x, rxn_class=rxn_node)
        edge_feat = self.bond_encoder(
            edge_feat=graphs.edge_attr, org_ptr=graphs.original_edge_ptr,
            pad_ptr=graphs.pad_edge_ptr, self_ptr=graphs.self_edge_ptr,
            rxn_class=rxn_edge
        )
        return node_feat, edge_feat

    def update_act_nodes(self, node_res, ptr, act_x=None):
        node_res = node_res.detach().cpu()
        node_res = torch.argmax(node_res, dim=-1)
        result = []
        for idx in range(len(ptr) - 1):
            node_res_t = node_res[ptr[idx]: ptr[idx + 1]]
            node_all = torch.arange(ptr[idx + 1] - ptr[idx])
            mask = node_res_t == 1
            t_result = set(node_all[mask].tolist())
            if act_x is not None:
                t_result |= set(act_x[idx])
            result.append(t_result)
        return result

    def forward(
        self, graphs, act_nodes=None, mode='together', e_types=None,
        empty_type=0, edge_map=None
    ):
        node_feat, edge_feat = self.get_init_feats(graphs)
        if self.sparse:
            node_feat, _ = self.base_model(
                node_feats=node_feat, edge_feats=edge_feat,
                edge_index=graphs.edge_index
            )
            edge_feat, node_res = None, self.node_predictor(node_feat)

        if mode in ['together', 'inference']:
            act_nodes = self.update_act_nodes(
                act_x=act_nodes if mode == 'together' else None,
                node_res=node_res, ptr=graphs.ptr
            )
        elif mode != 'original':
            raise NotImplementedError(f'Invalid mode: {mode}')

        pred_edge, e_answer, e_ptr = self.predict_edge(
            node_feat=node_feat, activate_nodes=act_nodes, edge_feat=edge_feat,
            e_types=e_types, empty_type=empty_type, edge_map=edge_map,
            ptr=graphs.ptr
        )
        return node_res, pred_edge, e_answer, e_ptr, act_nodes


def get_label(e_type, x, y, empty_type=0):
    if e_type is None:
        return empty_type
    if (x, y) in e_type:
        return e_type[(x, y)]
    elif (y, x) in e_type:
        return e_type[(y, x)]
    else:
        return empty_type


def evaluate_sparse(node_res, edge_res, e_labels, node_ptr, e_ptr, act_nodes):
    node_cover, node_fit, edge_fit, all_fit, all_cover = 0, 0, 0, 0, 0
    node_res = node_res.cpu().argmax(dim=-1)
    if edge_res is not None:
        edge_res = edge_res.cpu().argmax(dim=-1)
    for idx, a_node in enumerate(act_nodes):
        t_node_res = node_res[node_ptr[idx]: node_ptr[idx + 1]] == 1
        real_nodes = torch.zeros_like(t_node_res, dtype=bool)
        real_nodes[a_node] = True
        inters = torch.logical_and(real_nodes, t_node_res)
        nf = torch.all(real_nodes == t_node_res).item()
        nc = torch.all(real_nodes == inters).item()

        if edge_res is not None:
            t_edge_res = edge_res[e_ptr[idx]: e_ptr[idx + 1]]
            t_edge_labels = e_labels[e_ptr[idx]: e_ptr[idx + 1]]
            ef = torch.all(t_edge_res == t_edge_labels).item()
        else:
            ef = True

        node_fit += nf
        node_cover += nc
        edge_fit += ef
        all_fit += (nf & ef)
        all_cover += (nc & ef)
    return node_cover, node_fit, edge_fit, all_fit, all_cover, len(act_nodes)
