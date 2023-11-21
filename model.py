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


def make_memory_from_feat(node_feat, batch_mask):
    batch_size, max_node = batch_mask.shape
    memory = torch.zeros(batch_size, max_node, node_feat.shape[-1])
    memory = memory.to(node_feat.device)
    memory[batch_mask] = node_feat
    return memory, ~batch_mask


class SynthonPredictionModel(torch.nn.Module):
    def __init__(self, base_model, node_dim, edge_dim, dropout=0.1):
        super(SynthonPredictionModel, self).__init__()
        self.base_model = base_model
        self.edge_predictor = torch.nn.Sequential(
            torch.nn.Linear(edge_dim, edge_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(edge_dim, 3)
        )

        self.node_predictor = torch.nn.Sequential(
            torch.nn.Linear(node_dim, node_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(node_dim, 7)
        )

    def calc_loss(
        self, node_logits, node_label, edge_logits, edge_label,
        node_batch, edge_batch,
    ):
        max_node_batch = node_batch.max().item() + 1
        node_loss = torch.zeros(max_node_batch).to(node_logits)
        node_loss_src = cross_entropy(
            node_logits, node_label, reduction='none'
        )
        node_loss.scatter_add_(0, node_batch, node_loss_src)

        max_edge_batch = edge_batch.max().item() + 1
        edge_loss = torch.zeros(max_edge_batch).to(edge_logits)
        edge_loss_src = cross_entropy(
            edge_logits, edge_label, reduction='none',
        )
        edge_loss.scatter_add_(0, edge_batch, edge_loss_src)

        return node_loss.mean(), edge_loss.mean()

    def forward(self, graph, ret_loss=True, ret_feat=False):
        node_feat, edge_feat = self.base_model(graph)

        node_logits = self.node_predictor(node_feat)
        node_logits = node_logits.squeeze(dim=-1)

        edge_logits = self.edge_predictor(edge_feat)
        edge_logits = edge_logits.squeeze(dim=-1)

        if ret_loss:
            n_loss, e_loss = self.calc_loss(
                node_logits=node_logits, edge_logits=edge_logits,
                node_label=graph.node_label, node_batch=graph.batch,
                edge_label=graph.edge_label, edge_batch=graph.e_batch
            )

        answer = (node_logits, edge_logits)
        if ret_loss:
            answer += (n_loss, e_loss)
        if ret_feat:
            answer += (node_feat, edge_feat)
        return answer


class OverallModel(torch.nn.Module):
    def __init__(
        self, GNN, trans_enc, trans_dec,
        node_dim, edge_dim, use_sim=False,
    ):
        super(OverallModel, self).__init__()
        self.GNN, self.trans_enc, self.trans_dec = GNN, trans_enc, trans_dec
        self.syn_e_pred = torch.nn.Sequential(
            torch.nn.Linear(edge_dim, edge_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(edge_dim, 3)
        )
        self.syn_n_pred = torch.nn.Sequential(
            torch.nn.Linear(node_dim, node_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(node_dim, 7)
        )
        self.lg_activate = torch.nn.Sequential(
            torch.nn.Linear(node_dim, node_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(node_dim, 1)
        )
        self.conn_pred = torch.nn.Sequential(
            torch.nn.Linear(edge_dim + edge_dim, edge_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(edge_dim, 1)
        )
        self.reac_embedding = torch.nn.Parameter(torch.randn(1, 1, node_dim))
        self.prod_embedding = torch.nn.Parameter(torch.randn(1, 1, node_dim))
        self.emb_trans = torch.nn.Linear(node_dim + node_dim, node_dim)

    def add_type_emb(self, x, is_reac):
        batch_size, max_len = x.shape[:2]
        if is_reac:
            type_emb = self.reac_embedding.repeat(batch_size, max_len, 1)
        else:
            type_emb = self.prod_embedding.repeat(batch_size, max_len, 1)

        return self.emb_trans(torch.cat([x, type_emb], dim=-1)) + x

    def trans_enc_forward(self, pre_graph, word_emb, word_pad, graph_emb, graph_pad):
        word_emb = self.add_type_emb(word_emb, is_reac=True)
        graph_emb = self.add_type_emb(graph_emb, is_reac=False)

        if pre_graph:
            trans_input = torch.cat([word_emb, graph_emb], dim=1)
            memory_pad = torch.cat([word_pad, graph_pad], dim=1)
            memory = self.trans_enc(trans_input, key_padding_mask=memory_pad)
        else:
            memory = self.trans_enc(word_emb, key_padding_mask=word_pad)
            memory = torch.cat([memory, graph_pad], dim=1)
            memory_pad = torch.cat([word_pad, graph_pad], dim=1)
        return memory, memory_pad


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
                # print(type(org_graphs))

                paras = {
                    'graphs': [org_graphs[x] for x in valid_idx],
                    'activate_nodes': [ext_n_lb[x].cpu() for x in valid_idx],
                    'changed_edges': [ext_e_lb[x].cpu() for x in valid_idx],
                    'pad_num': pad_num, 'rxns': rxns,
                    'node_types': [
                        {idx: v.item() for idx, v in enumerate(sep_nodes[x])}
                        for x in valid_idx
                    ],
                    'edge_types': [sep_edges[x] for x in valid_idx]
                }

                aug_dec_G, aug_type = make_decoder_graph(**paras)
                aug_dec_G = aug_dec_G.to(enc_n_pred.device)
                a, b, c, d = self.decoder(
                    aug_dec_G, memory[valid_idx], aug_type,
                    mem_pad_mask=memory_pad_mask[valid_idx],
                    matching=use_matching,
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
        enc_n_pred = [x.cpu() for x in enc_n_pred]
        enc_e_pred = [x.cpu() for x in enc_e_pred]
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
        ).to(memory.device)

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
