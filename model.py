import torch
from sparse_backBone import (
    GINBase, GATBase, SparseAtomEncoder, SparseBondEncoder
)
from itertools import combinations, permutations
from torch_geometric.data import Data
from typing import Any, Dict, List, Tuple, Optional, Union
import numpy as np
from torch.nn.functional import binary_cross_entropy_with_logits


class BinaryGraphEditModel(torch.nn.Module):
    def __init__(self, base_model, node_dim, edge_dim, dropout=0.1):
        super(GraphEditModel, self).__init__()
        self.base_model = base_model

        self.edge_predictor = torch.nn.Sequential(
            torch.nn.Linear(edge_dim, edge_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(edge_dim, 1)
        )

        self.node_predictor = torch.nn.Sequential(
            torch.nn.Linear(node_dim, node_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(node_dim, 1)
        )

    def make_useful_mask(
        self, edge_index, mode, real_label=None, pred_label=None,
    ):
        def f(src, dst, lb):
            xa = torch.index_select(lb, dim=0, src) == 1
            xb = torch.index_select(lb, dim=0, dst) == 1
            return torch.logical_and(xa, xb)

        assert mode in ['inference', 'all', 'merge', 'org'], \
            f'Invalid mode {mode} for making mask found'
        num_edges, (src, dst) = edge_index.shape[1], edge_index
        if mode == 'all':
            useful_mask = torch.ones(num_edges).bool()
            useful_mask = useful_mask.to(edge_index.device)
        elif mode == 'merge':
            assert real_label is not None, f'Missing real for mode {mode}'
            assert pred_label is not None, f'Missing pred for mode {mode}'
            mix_label = torch.logical_or(real_label == 1, pred_label == 1)
            useful_mask = f(src, dst, mix_label)
        elif mode == 'org':
            assert real_label is not None, f'Missing real for mode {mode}'
            useful_mask = f(src, dst, real_label == 1)
        else:
            assert pred_label is not None, f'Missing pred for mode {mode}'
            useful_mask = f(src, dst, pred_label == 1)
        return useful_mask

    def calc_loss(
        self, node_logits, node_label, edge_logits, edge_label,
        reduction, graph_level, node_batch=None, edge_batch=None
    ):
        assert reduction in ['mean', 'sum'], \
            f'Invalid reduction method {reduction}'

        if not graph_level:
            node_loss = binary_cross_entropy_with_logits(
                node_logits, node_label, reduction=reduction
            )
            edge_loss = binary_cross_entropy_with_logits(
                edge_logits, edge_label, reduction=reduction
            )
            return node_loss, edge_loss
        else:
            assert node_batch is not None, 'require node_batch'
            assert edge_batch is not None, 'require edge_batch'
            max_node_batch = node_batch.max().item() + 1
            max_edge_batch = edge_batch.max().item() + 1
            node_loss = torch.zeros(max_node_batch).to(node_logits)
            edge_loss = torch.zeros(max_edge_batch).to(edge_logits)
            node_loss_src = binary_cross_entropy_with_logits(
                node_logits, node_label, reduction='none'
            )
            edge_loss_src = binary_cross_entropy_with_logits(
                edge_logits, edge_label, reduction='none'
            )

            node_loss.scatter_add_(0, node_batch, node_loss_src)
            edge_loss.scatter_add_(0, edge_batch, edge_loss_src)

            if reduction == 'mean':
                return node_loss.mean(), edge_loss.mean()
            else:
                return node_loss.sum(), edge_loss.sum()

    def forward(
        self, graph, mask_mode, reduce_mode,
        graph_level=True, ret_loss=True
    ):
        node_feat, edge_feat = self.base_model(graph)

        node_logits = self.node_predictor(node_feat)
        node_logits = node_logits.squeeze(dim=-1)

        node_pred = node_logits.detach().clone()
        node_pred[node_pred > 0] = 1
        node_pred[node_pred <= 0] = 0

        useful_mask = self.make_useful_mask(
            edge_index=graph.edge_index, mode=mask_mode,
            real_label=graph.node_label, pred_label=node_pred
        )

        edge_logits = self.edge_predictor(edge_feat[useful_mask])
        edge_logits = edge_logits.squeeze(dim=-1)

        edge_pred = edge_logits.detach().clone()
        edge_pred[edge_pred > 0] = 1
        edge_pred[edge_pred <= 0] = 0

        if ret_loss:
            n_loss, e_loss = self.calc_loss()
            return node_pred, edge_pred, n_loss, e_loss
        else:
            return node_pred, edge_pred


# def evaluate_sparse(node_res, edge_res, e_labels, node_ptr, e_ptr, act_nodes):
#     node_cover, node_fit, edge_fit, all_fit, all_cover = 0, 0, 0, 0, 0
#     node_res = node_res.cpu().argmax(dim=-1)
#     if edge_res is not None:
#         edge_res = edge_res.cpu().argmax(dim=-1)
#     for idx, a_node in enumerate(act_nodes):
#         t_node_res = node_res[node_ptr[idx]: node_ptr[idx + 1]] == 1
#         real_nodes = torch.zeros_like(t_node_res, dtype=bool)
#         real_nodes[a_node] = True
#         inters = torch.logical_and(real_nodes, t_node_res)
#         nf = torch.all(real_nodes == t_node_res).item()
#         nc = torch.all(real_nodes == inters).item()

#         if edge_res is not None:
#             t_edge_res = edge_res[e_ptr[idx]: e_ptr[idx + 1]]
#             t_edge_labels = e_labels[e_ptr[idx]: e_ptr[idx + 1]]
#             ef = torch.all(t_edge_res == t_edge_labels).item()
#         else:
#             ef = True

#         node_fit += nf
#         node_cover += nc
#         edge_fit += ef
#         all_fit += (nf & ef)
#         all_cover += (nc & ef)
#     return node_cover, node_fit, edge_fit, all_fit, all_cover, len(act_nodes)
