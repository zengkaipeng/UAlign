import torch
from sparse_backBone import (
    GINBase, GATBase, SparseAtomEncoder, SparseBondEncoder
)
from itertools import combinations, permutations
from torch_geometric.data import Data
from typing import Any, Dict, List, Tuple, Optional, Union
import numpy as np
from torch.nn.functional import binary_cross_entropy_with_logits
from torch.nn.functional import cross_entropy
from decoder import batch_mask, graph2batch


class BinaryGraphEditModel(torch.nn.Module):
    def __init__(self, base_model, node_dim, edge_dim, dropout=0.1):
        super(BinaryGraphEditModel, self).__init__()
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
            xa = torch.index_select(lb, dim=0, index=src) > 0
            xb = torch.index_select(lb, dim=0, index=dst) > 0
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
            mix_label = torch.logical_or(real_label > 0, pred_label > 0)
            useful_mask = f(src, dst, mix_label)
        elif mode == 'org':
            assert real_label is not None, f'Missing real for mode {mode}'
            useful_mask = f(src, dst, real_label > 0)
        else:
            assert pred_label is not None, f'Missing pred for mode {mode}'
            useful_mask = f(src, dst, pred_label > 0)
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
            if edge_logits.numel() > 0:
                edge_loss = binary_cross_entropy_with_logits(
                    edge_logits, edge_label, reduction=reduction
                )
            else:
                edge_loss = 0
            return node_loss, edge_loss
        else:
            assert node_batch is not None, 'require node_batch'
            assert edge_batch is not None, 'require edge_batch'
            max_node_batch = node_batch.max().item() + 1
            node_loss = torch.zeros(max_node_batch).to(node_logits)
            node_loss_src = binary_cross_entropy_with_logits(
                node_logits, node_label, reduction='none'
            )
            node_loss.scatter_add_(0, node_batch, node_loss_src)

            if edge_batch.numel() > 0:
                max_edge_batch = edge_batch.max().item() + 1
                edge_loss = torch.zeros(max_edge_batch).to(edge_logits)
                edge_loss_src = binary_cross_entropy_with_logits(
                    edge_logits, edge_label, reduction='none'
                )
                edge_loss.scatter_add_(0, edge_batch, edge_loss_src)
            else:
                edge_loss = torch.Tensor([0]).to(node_loss)

            if reduction == 'mean':
                return node_loss.mean(), edge_loss.mean()
            else:
                return node_loss.sum(), edge_loss.sum()

    def forward(
        self, graph, mask_mode, reduce_mode='mean',
        graph_level=True, ret_loss=True, ret_feat=False
    ):
        node_feat, edge_feat = self.base_model(graph)

        node_logits = self.node_predictor(node_feat)
        node_logits = node_logits.squeeze(dim=-1)

        node_pred = node_logits.detach().clone()
        node_pred[node_pred > 0] = 1
        node_pred[node_pred <= 0] = 0

        if mask_mode != 'inference':
            useful_mask = self.make_useful_mask(
                edge_index=graph.edge_index, mode=mask_mode,
                real_label=graph.node_label, pred_label=node_pred
            )
        else:
            useful_mask = self.make_useful_mask(
                graph.edge_index, mode=mask_mode,
                pred_label=node_pred
            )

        edge_logits = self.edge_predictor(edge_feat[useful_mask])
        edge_logits = edge_logits.squeeze(dim=-1)

        edge_pred = edge_logits.detach().clone()
        edge_pred[edge_pred > 0] = 1
        edge_pred[edge_pred <= 0] = 0

        if ret_loss:
            n_loss, e_loss = self.calc_loss(
                node_logits=node_logits, edge_logits=edge_logits,
                node_label=graph.node_label, node_batch=graph.batch,
                edge_label=graph.edge_label[useful_mask],
                edge_batch=graph.e_batch[useful_mask],
                reduction=reduce_mode, graph_level=graph_level
            )
            answer = (node_pred, edge_pred, useful_mask, n_loss, e_loss)
        else:
            answer = (node_pred, edge_pred, useful_mask)

        if ret_feat:
            answer += (node_feat, edge_feat)
        return answer


def make_ptr_from_batch(batch, batch_size=None):
    if batch_size is None:
        batch_size = batch.max().item() + 1
    ptr = torch.zeros(batch_size).to(batch)
    ptr.scatter_add_(dim=0, src=torch.ones_like(batch), index=batch)
    ptr = torch.cat([torch.Tensor([0]).to(batch.device), ptr], dim=0)
    ptr = torch.cumsum(ptr, dim=0).long()
    return ptr


def evaluate_sparse(
    node_pred, edge_pred, node_batch, edge_batch,
    node_label, edge_label, batch_size
):
    node_cover, node_fit, edge_fit, edge_cover, all_fit, all_cover = [0] * 6
    node_ptr = make_ptr_from_batch(node_batch, batch_size)
    if edge_batch.numel() > 0:
        edge_ptr = make_ptr_from_batch(edge_batch, batch_size)
    for idx in range(batch_size):
        t_node_res = node_pred[node_ptr[idx]: node_ptr[idx + 1]] > 0
        real_nodes = node_label[node_ptr[idx]: node_ptr[idx + 1]] > 0
        inters = torch.logical_and(t_node_res, real_nodes)
        nf = torch.all(real_nodes == t_node_res).item()
        nc = torch.all(real_nodes == inters).item()

        if edge_batch.numel() > 0:
            t_edge_res = edge_pred[edge_ptr[idx]: edge_ptr[idx + 1]] > 0
            real_edges = edge_label[edge_ptr[idx]: edge_ptr[idx + 1]] > 0
            e_inters = torch.logical_and(t_edge_res, real_edges)
            ef = torch.all(t_edge_res == real_edges).item()
            ec = torch.all(real_edges == e_inters).item()
        else:
            ef = ec = True

        node_fit += nf
        node_cover += nc
        edge_fit += ef
        edge_cover += ec
        all_fit += (nf & ef)
        all_cover += (nc & ec)
    return node_cover, node_fit, edge_cover, edge_fit, all_cover, all_fit


class DecoderOnly(torch.nn.Module):
    def __init__(
        self, backbone, node_dim, edge_dim,
        node_class, edge_class
    ):
        super(DecoderOnly, self).__init__()
        self.backbone = backbone
        self.node_predictor = torch.nn.Linear(node_dim, node_class)
        self.edge_predictor = torch.nn.Linear(edge_dim, edge_class)

    def forward(self, graph, memory, mem_pad_mask=None):
        node_feat, edge_feat = self.backbone(graph, memory, mem_pad_mask)
        node_pred = self.node_predictor(node_feat)
        edge_pred = self.edge_predictor(edge_feat)
        return node_pred, edge_pred


class EncoderDecoder(torch.nn.Module):
    def __init__(self, encoder, decoder):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(
        self, encoder_graph, decoder_graph, encoder_mode,
        reduction='mean', graph_level=True, ret_loss=True,
    ):
        encoder_answer = self.encoder(
            graph=encoder_graph, graph_level=graph_level,
            ret_loss=ret_loss, ret_feat=True, mode=encoder_mode,
            reduce_mode=reduction,
        )

        if ret_loss:
            node_pred, edge_pred, useful_mask, n_loss, e_loss, \
                node_feat, edge_feat = encoder_answer
        else:
            node_pred, edge_pred, useful_mask, node_feat, edge_feat\
                = encoder_answer

        device = encoder_graph.x.device
        batch_size = encoder_graph.batch.max().item() + 1
        n_nodes = torch.zeros(batch_size).long().to(device)
        n_nodes.scatter_add_(
            src=torch.ones_like(encoder_graph.batch),
            dim=0, index=encoder_graph.batch
        )
        max_node = n_nodes.max().item() + 1
        mem_pad_mask = batch_mask(encoder_graph.ptr, max_node, batch_size)

        memory = torch.zeros(batch_size, max_node, node_feat.shape[-1])
        memory = memory.to(device)
        memory[mem_pad_mask] = node_feat

        node_logits, edge_logits = self.decoder(
            graph=decoder_graph, memory=memory,
            mem_pad_mask=mem_pad_mask
        )

    def loss_calc(
        self, node_logits, edge_logits, pad_node_label, edge_type_dict,
        graph, reduction='mean', graph_level=True, alpha=1
    ):
        """[summary]

        calc the loss for decoder with bin matching

        Args:
            node_logits ([type]): A tensor of (N nodes, n_class), the prediction
                for the whole batch nodes
            edge_logits ([type]): A tensor of (N edge, e_class), the prediction
                for the whole batch edge
            org_node_label ([type]): A tensor of (N node_org), the label 
                of all the original nodes
            org_edge_label ([type]): A tensor of (N edge_org), the label for 
                all the original edges 
            pad_node_label ([type]): A tesnor of (batch_size, N_pad), the label
                for all the padding nodes
            edge_type_dict ([type]): A dict containing all the edge_type of 
                every edges in the batch, in form of (idx_i, idx_j) -> e_type
            reduce (str): the reduction method of loss (default: `'mean'`)
            graph_level (bool): reduce all the losses first to graph 
                then to the other, (default: `True`)
            alpha (number): the weight of org part, as aux loss (default: `1`)
        """
        batch_size = pad_node_label.shape[0]
        device = node_logits.device
        org_node_logits = node_logits[graph.node_org_mask]

        org_edge_logits = edge_logits[graph.org_mask]

        if self.graph_level:
            org_node_loss = cross_entropy(
                org_node_logits, graph.org_node_labels, reduction='none'
            )
            org_edge_loss = cross_entropy(
                org_edge_logits, graph.org_edge_labels, reduction='none'
            )
            org_node_batch = graph.batch[graph.node_org_mask]
            org_edge_batch = graph.e_batch[graph.org_mask]
            org_node_loss_graph = torch.zeros(batch_size).to(device)
            org_edge_loss_graph = torch.zeros(batch_size).to(device)

            org_node_loss_graph.scatter_add_(
                dim=0, src=org_node_loss, index=org_node_batch
            )
            org_edge_loss_graph.scatter_add_(
                dim=0, src=org_edge_loss, index=org_edge_batch
            )

            if reduction == 'mean':
                org_node_loss = org_node_loss_graph.mean()
                org_edge_loss = org_edge_loss_graph.mean()
            else:
                org_node_loss = org_node_loss_graph.sum()
                org_edge_loss = org_edge_loss_graph.sum()

        else:
            org_node_loss = cross_entropy(
                org_node_logits, graph.org_node_labels, reduction=reduction
            )
            org_edge_loss = cross_entropy(
                org_edge_logits, graph.org_edge_labels, reduction=reduction
            )
