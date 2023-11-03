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
    seperate_dict, extend_label_by_edge, filter_label_by_node,
    seperate_encoder_graphs, seperate_pred
)
from Dataset import make_decoder_graph


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
        node_class, edge_class, pad_num
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
        self.pad_num = pad_num

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
            org_n_cls=graph.node_class[graph.n_org_mask],
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

        return org_n_loss, org_e_loss, pad_n_loss, pad_e_loss

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

        pad_e_pred = {k: v for k, v in pad_e_pred.items() if v != 0}

        edge_res = seperate_dict(
            label_dict=pad_e_pred, num_nodes=graph.num_nodes,
            batch=graph.batch, ptr=graph.ptr
        )
        return node_res, edge_res

    def calc_org_loss(
        self, batch_size, node_batch, edge_batch,
        org_n_logs, org_n_cls, org_e_logs, org_e_cls
    ):
        node_loss = torch.zeros(batch_size).to(org_n_logs)
        org_node_loss = cross_entropy(
            org_n_logs, org_n_cls, reduction='none'
        )
        node_loss.scatter_add_(0, node_batch, org_node_loss)

        edge_loss = torch.zeros(batch_size).to(org_e_logs)
        org_edge_loss = cross_entropy(
            org_e_logs, org_e_cls, reduction='none'
        )
        edge_loss.scatter_add_(0, edge_batch, org_edge_loss)

        return node_loss.mean(), edge_loss.mean()

    def get_matching(self, pad_logits, pad_cls):
        neg_prob = -torch.log_softmax(pad_logits, dim=-1)
        val_matrix = neg_prob[:, pad_cls]
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
            this_edges = pad_e_index[:, pad_e_batch == idx]
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
                row_id = col_id = list(range(self.pad_num))
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
        self, encoder_graph, decoder_graph, edge_types,
        pos_weight=1, use_matching=True, aug_mode='none'
    ):
        enc_n_log, enc_e_log, enc_n_loss, enc_e_loss, n_feat, e_feat =\
            self.encoder(encoder_graph, True, True, pos_weight)

        memory, memory_pad_mask = make_memory_from_feat(
            node_feat=n_feat, batch_mask=encoder_graph.batch_mask
        )
        org_n_loss, org_e_loss, pad_n_loss, pad_e_loss = self.decoder(
            decoder_graph, memory, edge_types, matching=use_matching,
            mem_pad_mask=memory_pad_mask
        )
        batch_size = memory.shape[0]

        if aug_mode != 'none':
            pad_num = self.decoder.pad_num
            enc_n_pred = convert_log_into_label(enc_n_log)
            enc_e_pred = convert_edge_log_into_labels(
                enc_e_log, encoder_graph.edge_index,
                mod='sigmoid', return_dict=False,
            )

            if aug_mode == 'node':
                enc_n_pred, enc_e_pred = filter_label_by_node(
                    enc_n_pred, enc_e_pred, encoder_graph.edge_index
                )
            elif aug_mode == 'edge':
                enc_n_pred, enc_e_pred = extend_label_by_edge(
                    enc_n_pred, enc_e_pred, encoder_graph.edge_index
                )
            else:
                raise NotImplementedError(f'Invalid aug_mode {aug_mode}')

            valid_idx = filter_valid_idx(
                enc_n_pred, enc_e_pred, encoder_graph.node_label,
                encoder_graph.edge_label, encoder_graph.batch,
                encoder_graph.e_batch, batch_size
            )

            if len(valid_idx) > 0:
                org_graphs = seperate_encoder_graphs(encoder_graph)
                if len(org_graphs) == 2:
                    org_graphs, rxns = org_graphs
                else:
                    rxns = None

                ext_n_lb = seperate_pred(
                    enc_n_pred, batch_size, encoder_graph.batch
                )
                ext_e_lb = seperate_pred(
                    enc_e_pred, batch_size, encoder_graph.e_batch
                )

                sep_edges = seperate_dict(
                    edge_types, decoder_graph.num_nodes,
                    decoder_graph.batch, decoder_graph.ptr
                )

                sep_nodes = seperate_pred(
                    decoder_graph.node_class, batch_size,
                    decoder_graph.batch
                )

                paras = {
                    'graphs': [org_graphs[x] for x in valid_idx],
                    'activate_nodes': [ext_n_lb[x] for x in valid_idx],
                    'changed_edges': [ext_e_lb[x] for x in valid_idx],
                    'pad_num': pad_num, 'rxns': rxns,
                    'node_types': [
                        {idx: v.item() for idx, v in enumerate(sep_nodes[x])}
                        for x in valid_idx
                    ],
                    'edge_types': [sep_edges[x] for x in valid_idx]
                }

                aug_dec_G = make_decoder_graph(**paras)
                a, b, c, d = self.decoder(
                    decoder_graph, memory, edge_types, matching=use_matching,
                    mem_pad_mask=memory_pad_mask
                )

                org_n_loss += a
                org_e_loss += b
                pad_n_loss += c
                pad_e_loss += d

        return enc_n_loss, enc_e_loss, org_n_loss, org_e_loss, pad_n_loss, pad_e_loss

    def predict(self, graph, syn_mode='edge'):
        enc_n_log, enc_e_log, n_feat, e_feat = self.encoder(
            graph, ret_feat=True, ret_loss=False
        )

        memory, mem_pad_mask = make_memory_from_feat(
            n_feat, graph.batch_mask
        )

        enc_n_pred = convert_log_into_label(enc_n_log)
        enc_e_pred = convert_edge_log_into_labels(
            enc_e_log, graph.edge_index, mod='sigmoid'
        )

        if syn_mode == 'node':
            enc_n_pred, enc_e_pred = filter_label_by_node(
                enc_n_pred, enc_e_pred, graph.edge_index
            )
        elif syn_mode == 'edge':
            enc_n_pred, enc_e_pred = extend_label_by_edge(
                enc_n_pred, enc_e_pred, graph.edge_index
            )
        else:
            raise NotImplementedError(f'Invalid aug_mode {syn_mode}')

        pad_num = self.decoder.pad_num
        batch_size = memory.shape[0]

        enc_n_pred = seperate_pred(enc_n_pred, batch_size, graph.batch)
        enc_e_pred = seperate_pred(enc_e_pred, batch_size, graph.e_batch)
        org_graphs = seperate_encoder_graphs(graph)
        if len(org_graphs) == 2:
            org_graphs, rxns = org_graphs
        else:
            rxns = None

        # print([x['num_nodes'] for x in org_graphs])
        # print(org_graphs[0])
        # print(org_graphs[1])

        decoder_graph = make_decoder_graph(
            org_graphs, enc_n_pred, enc_e_pred, pad_num, rxns=rxns
        )

        pad_n_pred, pad_e_pred = self.decoder.predict(
            decoder_graph, memory, mem_pad_mask
        )

        return enc_n_pred, enc_e_pred, pad_n_pred, pad_e_pred


def filter_valid_idx(
    node_pred, edge_pred, n_lb, e_lb, batch, e_batch, batch_size
):
    answer = []
    for idx in range(batch_size):
        this_n_mask = batch == idx
        this_e_mask = e_batch == idx

        this_nlb = n_lb[this_n_mask] > 0
        this_elb = e_lb[this_e_mask] > 0

        this_npd = node_pred[this_n_mask] > 0
        this_epd = edge_pred[this_e_mask] > 0

        inters = this_nlb & this_npd
        nc = torch.all(this_nlb == inters).item()

        inters = this_elb & this_epd
        ec = torch.all(this_elb == inters).item()
        if nc and ec:
            answer.append(idx)
    return answer
