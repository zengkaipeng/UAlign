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
from scipy.optimize import linear_sum_assignment
from data_utils import (
    convert_log_into_label, convert_edge_log_into_labels,
    seperate_dict
)


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

    def calc_loss(
        self, node_logits, node_label, edge_logits, edge_label,
        node_batch, edge_batch, pos_weight=1
    ):
        max_node_batch = node_batch.max().item() + 1
        node_loss = torch.zeros(max_node_batch).to(node_logits)
        node_loss_src = binary_cross_entropy_with_logits(
            node_logits, node_label, reduction='none',
            pos_weight=torch.tensor(pos_weight)
        )
        node_loss.scatter_add_(0, node_batch, node_loss_src)

        max_edge_batch = edge_batch.max().item() + 1
        edge_loss = torch.zeros(max_edge_batch).to(edge_logits)
        edge_loss_src = binary_cross_entropy_with_logits(
            edge_logits, edge_label, reduction='none',
            pos_weight=torch.tensor(pos_weight)
        )
        edge_loss.scatter_add_(0, edge_batch, edge_loss_src)

        return node_loss.mean(), edge_loss.mean()

    def forward(self, graph, ret_loss=True, ret_feat=False, pos_weight=1):
        node_feat, edge_feat = self.base_model(graph)

        node_logits = self.node_predictor(node_feat)
        node_logits = node_logits.squeeze(dim=-1)

        edge_logits = self.edge_predictor(edge_feat)
        edge_logits = edge_logits.squeeze(dim=-1)

        if ret_loss:
            n_loss, e_loss = self.calc_loss(
                node_logits=node_logits, edge_logits=edge_logits,
                node_label=graph.node_label, node_batch=graph.batch,
                edge_label=graph.edge_label, edge_batch=graph.e_batch,
                pos_weight=pos_weight
            )

        answer = (node_logits, edge_logits)
        if ret_loss:
            answer += (n_loss, e_loss)
        if ret_feat:
            answer += (node_feat, edge_feat)
        return answer

    def make_memory(self, graph):
        node_feat, edge_feat = self.base_model(graph)
        return make_memory_from_feat(node_feat, graph.batch_mask)


def make_memory_from_feat(node_feat, batch_mask):
    batch_size, max_node = batch_mask.shape
    memory = torch.zeros(batch_size, max_node, node_feat.shape[-1])
    memory = memory.to(node_feat.device)
    memory[batch_mask] = node_feat
    return memory, ~batch_mask


class DecoderOnly(torch.nn.Module):
    def __init__(
        self, backbone, node_dim, edge_dim,
        node_class, edge_class
    ):
        super(DecoderOnly, self).__init__()
        self.backbone = backbone
        self.node_predictor = torch.nn.Sequential(
            torch.nn.Linear(node_dim, node_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(node_dim, node_class)
        )
        self.edge_predictor = torch.nn.Sequential(
            torch.nn.Linear(node_dim, node_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(node_dim, edge_class)
        )
        self.feat_extracter = torch.nn.Sequential(
            torch.nn.Linear(node_dim * 2, edge_dim),
            torch.nn.ReLU()
        )
        self.node_class = node_class
        self.edge_class = edge_class

    def forward(
        self, graph, memory, all_edge_types,
        mem_pad_mask=None, matching=True
    ):
        node_feat, edge_feat = self.backbone(graph, memory, mem_pad_mask)
        node_logits = self.node_predictor(node_feat)
        org_edge_logits = self.edge_predictor(edge_feat)
        device = node_logits.device
        batch_size = graph.batch.max().item() + 1
        all_node_index = torch.arange(graph.num_nodes).to(device)

        org_n_loss, org_e_loss = self.calc_org_loss(
            batch_size=batch_size,
            node_batch=graph.batch[graph.n_org_mask],
            edge_batch=graph.e_batch[graph.e_org_mask],
            org_n_logs=node_logits[graph.n_org_mask],
            ord_n_cls=graph.node_class[graph.n_org_mask],
            org_e_logs=org_edge_logits,
            org_e_cls=graph.org_edge_class
        )
        pad_n_loss, pad_e_loss = self.calc_pad_loss(
            batch_size=batch_size, node_feat=node_feat,
            pad_n_logs=node_logits[graph.n_pad_mask],
            pad_n_cls=graph.node_class[graph.n_pad_mask],
            pad_n_idx=all_node_index[graph.n_pad_mask],
            pad_e_index=graph.edge_index[:, graph.e_pad_mask],
            pad_e_batch=graph.e_batch[graph.e_pad_mask],
            all_edge_types=all_edge_types, use_matching=matching
        )

    def predict(self, graph, memory, mem_pad_mask=None):
        node_feat, _ = self.backbone(graph, memory, mem_pad_mask)
        node_logits = self.node_predictor(node_feat)
        device = node_logits.device
        batch_size = graph.batch.max().item() + 1
        all_node_index = torch.arange(graph.num_nodes).to(device)

        node_pred = convert_log_into_label(node_logits, mod='softmax')
        org_node_index = all_node_index - graph.ptr[graph.batch]

        # solving nodes
        pad_n_pred = node_pred[graph.n_pad_mask]
        pad_n_idx = org_node_index[graph.n_pad_mask]

        pad_n_pred = pad_n_pred.reshape(batch_size, -1)
        pad_n_idx = pad_n_idx.reshape(batch_size, -1)

        node_res = [{
            pad_n_idx[idx][i].item(): v.item()
            for i, v in enumerate(pad_n_pred[idx])
        } for idx in range(batch_size)]

        useful_node = (node_pred != 0) | graph.n_org_mask
        # unpadded nodes and nodes are not None are useful
        useful_edge = graph.e_pad_mask
        useful_edge &= useful_node[graph.edge_index[0]]
        useful_edge &= useful_node[graph.edge_index[1]]
        # padded edges between useful nodes are useful

        row, col = graph.edge_index[:, useful_edge]
        e_feat = torch.cat([node_feat[row], node_feat[col]], dim=-1)
        e_feat = self.feat_extracter(e_feat)
        edge_logits = self.edge_predictor(e_feat)
        pad_e_pred = convert_edge_log_into_labels(
            edge_logits, graph.edge_index[:, useful_edge],
            mod='softmax', return_dict=True
        )

        edge_res = seperate_dict(
            label_dict=pad_n_pred, num_nodes=graph.num_nodes,
            batch=graph.batch, ptr=graph.ptr
        )
        return node_res, edge_res

    def calc_org_loss(
        self, batch_size, node_batch, edge_batch,
        org_n_logs, ord_n_cls, org_e_logs, org_e_cls
    ):
        node_loss = torch.zeros(batch_size).to(node_logits)
        org_node_loss = cross_entropy(
            org_n_logs, org_node_cls, reduction='none'
        )
        node_loss.scatter_add_(0, node_batch, node_loss_src)

        edge_loss = torch.zeros(batch_size).to(edge_logits)
        org_edge_loss = cross_entropy(
            org_e_logs, org_e_cls, reduction='none'
        )
        edge_loss.scatter_add_(0, edge_batch, edge_loss_src)

        return node_loss.mean(), edge_loss.mean()

    def get_matching(self, pad_logits, pad_cls):
        neg_prob = -torch.softmax(pad_logits, dim=-1)
        val_matrix = neg_node_log_prob[:, pad_node_label]
        val_matrix = val_matrix.cpu().numpy()
        row_id, col_id = linear_sum_assignment(val_matrix)
        return row_id, col_id

    def calc_pad_loss(
        self, batch_size, pad_n_logs, pad_n_cls, node_feat,
        pad_n_idx, pad_e_index, pad_e_batch, all_edge_types,
        use_matching
    ):
        pad_n_logs = pad_n_logs.reshape(batch_size, -1, self.node_class)
        pad_n_cls = pad_n_cls.reshape(batch_size, -1)
        pad_n_idx = pad_n_idx.reshape(batch_size, -1)
        device = pad_n_logs.device

        total_n_loss = torch.zeros(batch_size).to(device)
        total_e_loss = torch.zeros(batch_size).to(device)
        for idx in range(batch_size):
            this_edges = pad_e_index[pad_e_batch == idx]
            if use_matching:
                with torch.no_grad():
                    row_id, col_id = self.get_matching(
                        pad_logits=pad_n_logs[idx],
                        pad_cls=pad_n_cls[idx]
                    )
                node_remap = {
                    pad_n_idx[idx][x].item(): pad_n_idx[idx][col_id[i]].item()
                    for i, x in enumerate(row_id)
                }
            else:
                node_remap = {}

            total_n_loss[idx] = cross_entropy(
                pad_n_logs[idx][row_id], pad_n_cls[idx][col_id],
                reduction='sum'
            )

            useless_nodes = set(
                pad_n_idx[idx][i].item() for i, v in
                enumerate(pad_n_cls[idx]) if v.item() == 0
            )

            useful_edges, e_labs = [], []
            for ex, (row, col) in enumerate(this_edges.T):
                x, y = row.item(), col.item()
                row = node_remap.get(x, x)
                col = node_remap.get(y, y)
                if row in useless_nodes or col in useless_nodes:
                    continue
                useful_edges.append((x, y))
                e_labs.append(all_edge_types.get((row, col), 0))

            useful_edges = torch.LongTensor(useful_edges).to(device)
            e_labs = torch.LongTensor(e_labs).to(device)

            idx_i, idx_j = useful_edges[:, 0], useful_edges[:, 1]
            e_feat = torch.cat([node_feat[idx_i], node_feat[idx_j]], dim=-1)
            e_logs = self.edge_predictor(self.feat_extracter(e_feat))

            total_e_loss[idx] = cross_entropy(e_logs, e_labs, reduction='sum')
        return total_n_loss.mean(), total_e_loss.mean()


class EncoderDecoder(torch.nn.Module):
    def __init__(self, encoder, decoder):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(
        self, encoder_graph, decoder_graph, encoder_mode, edge_types,
        reduction='mean', graph_level=True, alpha=1
    ):
        encoder_answer = self.encoder(
            graph=encoder_graph, graph_level=graph_level,
            mask_mode=encoder_mode, ret_loss=True,
            ret_feat=True, reduce_mode=reduction,
        )

        node_pred, edge_pred, useful_mask, n_loss, e_loss, \
            node_feat, edge_feat = encoder_answer

        device = encoder_graph.x.device
        batch_size = encoder_graph.batch.max().item() + 1
        n_nodes = torch.zeros(batch_size).long().to(device)
        n_nodes.scatter_add_(
            src=torch.ones_like(encoder_graph.batch),
            dim=0, index=encoder_graph.batch
        )
        max_node = n_nodes.max().item() + 1
        mem_pad_mask = make_batch_mask(encoder_graph.ptr, max_node, batch_size)

        memory = torch.zeros(batch_size, max_node, node_feat.shape[-1])
        memory = memory.to(device)
        memory[mem_pad_mask] = node_feat

        node_logits, edge_logits = self.decoder(
            graph=decoder_graph, memory=memory,
            mem_pad_mask=mem_pad_mask
        )

        decoder_losses = self.loss_calc(
            graph=decoder_graph, node_logits=node_logits,
            reduction=reduction, edge_logits=edge_logits,
            edge_type_dict=edge_types, graph_level=graph_level
        )
        org_node_loss, org_edge_loss, x_loss, c_loss = decoder_losses

        return n_loss + e_loss + x_loss + c_loss \
            + (org_node_loss + org_edge_loss) * alpha
