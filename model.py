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
import math
from Mix_backbone import DotMhAttn


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
            torch.nn.Linear(edge_dim, 5)
        )
        self.Echange = torch.nn.Sequential(
            torch.nn.Linear(node_dim, node_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(node_dim, 1)
        )
        self.Hchange = torch.nn.Sequential(
            torch.nn.Linear(node_dim, node_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(node_dim, 1)
        )
        self.Cchange = torch.nn.Sequential(
            torch.nn.Linear(node_dim, node_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(node_dim, 1)
        )

    def calc_loss(
        self, edge_logits, edge_label, edge_batch, AE_log, AE_label,
        AC_log, AC_label, AH_log, AH_label, batch,
    ):
        edge_loss = self.scatter_loss_by_batch(
            edge_logits, edge_label, edge_batch, cross_entropy
        )

        AC_loss = self.scatter_loss_by_batch(
            AC_log, AC_label, batch, binary_cross_entropy_with_logits
        )

        AH_loss = self.scatter_loss_by_batch(
            AH_log, AH_label, batch, binary_cross_entropy_with_logits
        )

        AE_loss = self.scatter_loss_by_batch(
            AE_log, AE_label, batch, binary_cross_entropy_with_logits
        )
        return edge_loss, AE_loss, AH_loss, AC_loss

    def scatter_loss_by_batch(self, logits, label, batch, lfn):
        max_batch = batch.max().item() + 1
        losses = torch.zeros(max_batch).to(logits)
        org_loss = lfn(logits, label, reduction='none')
        losses.index_add_(0, batch, org_loss)
        return losses.mean()

    def forward(self, graph):
        node_feat, edge_feat = self.base_model(graph)
        edge_logits = self.edge_predictor(edge_feat)

        AE_logits = self.Echange(node_feat).squeeze(dim=-1)
        AH_logits = self.Hchange(node_feat).squeeze(dim=-1)
        AC_logits = self.Cchange(node_feat).squeeze(dim=-1)

        e_loss, AE_loss, AH_loss, AC_loss = self.calc_loss(
            edge_logits=edge_logits, edge_label=graph.new_edge_types,
            edge_batch=graph.e_batch, AE_log=AE_logits,
            AE_label=graph.EdgeChange, AC_log=AC_logits,
            AC_label=graph.ChargeChange, AH_log=AH_logits,
            AH_label=graph.HChange, batch=graph.batch,
        )

        return e_loss, AE_loss, AH_loss, AC_loss

    def eval_forward(self, graph, return_all=False):
        node_feat, edge_feat = self.base_model(graph)
        edge_logits = self.edge_predictor(edge_feat)
        if return_all:
            AE_logits = self.Echange(node_feat).squeeze(dim=-1)
            AH_logits = self.Hchange(node_feat).squeeze(dim=-1)
            AC_logits = self.Cchange(node_feat).squeeze(dim=-1)

            return edge_logits, AE_logits, AH_logits, AC_logits
        else:
            return edge_logits


class PositionalEncoding(torch.nn.Module):
    def __init__(self, emb_size: int, dropout: float, maxlen: int = 2000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(
            - torch.arange(0, emb_size, 2) * math.log(10000) / emb_size
        )
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)

        self.dropout = torch.nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: torch.Tensor):
        token_len = token_embedding.shape[1]
        return self.dropout(token_embedding + self.pos_embedding[:token_len])


class OverallModel(torch.nn.Module):
    def __init__(
        self, GNN, trans_enc, trans_dec, node_dim, edge_dim, num_token,
        use_sim=True, pre_graph=True, heads=1, dropout=0.0, maxlen=2000,
        rxn_num=None
    ):
        super(OverallModel, self).__init__()
        self.GNN, self.trans_enc, self.trans_dec = GNN, trans_enc, trans_dec
        self.syn_e_pred = torch.nn.Sequential(
            torch.nn.Linear(edge_dim, edge_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(edge_dim, 5)
        )
        self.Echange = torch.nn.Sequential(
            torch.nn.Linear(node_dim, node_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(node_dim, 1)
        )
        self.Hchange = torch.nn.Sequential(
            torch.nn.Linear(node_dim, node_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(node_dim, 1)
        )
        self.Cchange = torch.nn.Sequential(
            torch.nn.Linear(node_dim, node_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(node_dim, 1)
        )
        self.lg_activate = torch.nn.Sequential(
            torch.nn.Linear(node_dim, node_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(node_dim, 1)
        )
        self.conn_pred = torch.nn.Sequential(
            torch.nn.Linear(node_dim + node_dim, node_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(node_dim, 4)
        )
        if rxn_num is None:
            self.tpp_embs = torch.nn.ParameterDict({
                'reac': torch.nn.Parameter(torch.randn(1, 1, node_dim)),
                'prod': torch.nn.Parameter(torch.randn(1, 1, node_dim))
            })
        else:
            self.tpp_embs = torch.nn.ModuleDict({
                'reac': torch.nn.Embedding(rxn_num, node_dim),
                'prod': torch.nn.Embedding(rxn_num, node_dim)
            })
        self.rxn_num = rxn_num
        self.emb_trans = torch.nn.Linear(node_dim + node_dim, node_dim)
        self.use_sim, self.pre_graph = use_sim, pre_graph
        if self.use_sim:
            self.SIM_G = SIM(node_dim, node_dim, heads, dropout)
            self.SIM_L = SIM(node_dim, node_dim, heads, dropout)

        self.token_embeddings = torch.nn.Embedding(num_token, node_dim)
        self.PE = PositionalEncoding(node_dim, dropout, maxlen)
        self.trans_pred = torch.nn.Linear(node_dim, num_token)

    def add_type_emb(self, x, part, graph_rxn=None):
        batch_size, max_len = x.shape[:2]
        if self.rxn_num is None:
            type_emb = self.tpp_embs[part].repeat(batch_size, max_len, 1)
        else:
            type_emb = self.tpp_embs[part](graph_rxn)

        return self.emb_trans(torch.cat([x, type_emb], dim=-1)) + x

    def trans_enc_forward(
        self, word_emb, word_pad, graph_emb, graph_pad,
        graph_rxn=None
    ):
        word_emb = self.add_type_emb(word_emb, 'reac', graph_rxn)
        word_emb = self.PE(word_emb)
        graph_emb = self.add_type_emb(graph_emb, 'prod', graph_rxn)

        if self.pre_graph:
            trans_input = torch.cat([word_emb, graph_emb], dim=1)
            memory_pad = torch.cat([word_pad, graph_pad], dim=1)
            memory = self.trans_enc(
                trans_input, src_key_padding_mask=memory_pad
            )
        else:
            memory = self.trans_enc(word_emb, src_key_padding_mask=word_pad)
            memory = torch.cat([memory, graph_emb], dim=1)
            memory_pad = torch.cat([word_pad, graph_pad], dim=1)
        return memory, memory_pad

    def conn_forward(self, lg_emb, graph_emb, conn_edges, node_mask):
        useful_edges_mask = node_mask[conn_edges[:, 1]]
        useful_src, useful_dst = conn_edges[useful_edges_mask].T
        conn_embs = [graph_emb[useful_src], lg_emb[useful_dst]]
        conn_embs = torch.cat(conn_embs, dim=-1)
        conn_logits = self.conn_pred(conn_embs)

        return conn_logits, useful_edges_mask

    def update_via_sim(self, graph_emb, graph_mask, lg_emb, lg_mask):
        graph_emb, g_pad_mask = make_memory_from_feat(graph_emb, graph_mask)
        lg_emb, l_pad_mask = make_memory_from_feat(lg_emb, lg_mask)

        new_graph_emb = self.SIM_G(graph_emb, lg_emb, l_pad_mask)
        new_lg_emb = self.SIM_L(lg_emb, graph_emb, g_pad_mask)
        return new_graph_emb[graph_mask], new_lg_emb[lg_mask]

    def forward(
        self, prod_graph, lg_graph, trans_ip, conn_edges, conn_batch,
        trans_op, graph_rxn=None, pad_idx=None, trans_ip_key_padding=None,
        trans_op_key_padding=None, trans_op_mask=None, trans_label=None,
        conn_label=None, mode='train', return_loss=False
    ):
        prod_n_emb, prod_e_emb = self.GNN(prod_graph)
        lg_n_emb, lg_e_emb = self.GNN(lg_graph)

        AE_logits = self.Echange(prod_n_emb).squeeze(dim=-1)
        AH_logits = self.Hchange(prod_n_emb).squeeze(dim=-1)
        AC_logits = self.Cchange(prod_n_emb).squeeze(dim=-1)
        prod_e_logits = self.syn_e_pred(prod_e_emb)

        trans_ip = self.token_embeddings(trans_ip)
        trans_op = self.token_embeddings(trans_op)

        batched_prod_emb, prod_padding_mask = \
            make_memory_from_feat(prod_n_emb, prod_graph.batch_mask)
        memory, memory_pad = self.trans_enc_forward(
            trans_ip, trans_ip_key_padding, batched_prod_emb,
            prod_padding_mask, graph_rxn
        )

        trans_pred = self.trans_pred(self.trans_dec(
            tgt=self.PE(trans_op), memory=memory, tgt_mask=trans_op_mask,
            memory_key_padding_mask=memory_pad,
            tgt_key_padding_mask=trans_op_key_padding
        ))

        lg_act_logits = self.lg_activate(lg_n_emb).squeeze(dim=-1)
        lg_useful = lg_graph.node_label > 0

        if self.use_sim:
            n_prod_emb, n_lg_emb = self.update_via_sim(
                prod_n_emb, prod_graph.batch_mask,
                lg_n_emb, lg_graph.batch_mask
            )
        else:
            n_prod_emb, n_lg_emb = prod_n_emb, lg_n_emb
        conn_logits, conn_mask = self.conn_forward(
            n_lg_emb, n_prod_emb,  conn_edges, lg_useful
        )

        if mode == 'train' or return_loss:
            losses = self.loss_calc(
                prod_e_log=prod_e_logits,
                prod_e_label=prod_graph.edge_label,
                AC_label=prod_graph.ChargeChange,
                AC_log=AC_logits,
                AH_label=prod_graph.HChange,
                AH_log=AH_logits,
                AE_label=prod_graph.EdgeChange,
                AE_log=AE_logits,
                prod_n_batch=prod_graph.batch,
                prod_e_batch=prod_graph.e_batch,
                lg_n_log=lg_act_logits,
                lg_n_label=lg_graph.node_label,
                lg_n_batch=lg_graph.batch,
                conn_lg=conn_logits,
                conn_lb=conn_label[conn_mask],
                conn_batch=conn_batch[conn_mask],
                trans_pred=trans_pred,
                trans_lb=trans_label,
                pad_idx=pad_idx
            )
        if mode == 'train':
            return losses
        else:
            answer = (
                AE_logits, AH_logits, AC_logits, prod_e_logits,
                lg_act_logits, conn_logits, conn_mask, trans_pred
            )
            return (answer, losses) if return_loss else answer

    def loss_calc(
        self, prod_e_log, prod_e_label, AC_log, AC_label,
        AH_log, AH_label, AE_log, AE_label,
        prod_n_batch, prod_e_batch, lg_n_log, lg_n_label, lg_n_batch,
        conn_lg, conn_lb, conn_batch, trans_pred, trans_lb, pad_idx
    ):
        AC_loss = self.scatter_loss_by_batch(
            AC_log, AC_label, prod_n_batch,
            binary_cross_entropy_with_logits
        )

        AH_loss = self.scatter_loss_by_batch(
            AH_log, AH_label, prod_n_batch,
            binary_cross_entropy_with_logits
        )

        AE_loss = self.scatter_loss_by_batch(
            AE_log, AE_label, prod_n_batch,
            binary_cross_entropy_with_logits
        )
        syn_edge_loss = self.scatter_loss_by_batch(
            prod_e_log, prod_e_label, prod_e_batch, cross_entropy
        )

        lg_act_loss = self.scatter_loss_by_batch(
            lg_n_log, lg_n_label, lg_n_batch,
            binary_cross_entropy_with_logits
        )

        conn_loss = self.scatter_loss_by_batch(
            conn_lg, conn_lb, conn_batch, cross_entropy
        )

        trans_loss = self.calc_trans_loss(trans_pred, trans_lb, pad_idx)

        return AC_loss, AE_loss, AH_loss,  syn_edge_loss, \
            lg_act_loss, conn_loss, trans_loss

    def scatter_loss_by_batch(self, logits, label, batch, lfn):
        max_batch = batch.max().item() + 1
        losses = torch.zeros(max_batch).to(logits)
        org_loss = lfn(logits, label, reduction='none')
        losses.index_add_(0, batch, org_loss)
        return losses.mean()

    def calc_trans_loss(self, trans_pred, trans_lb, ignore_index):
        batch_size, maxl, num_c = trans_pred.shape
        trans_pred = trans_pred.reshape(-1, num_c)
        trans_lb = trans_lb.reshape(-1)

        losses = cross_entropy(
            trans_pred, trans_lb, reduction='none',
            ignore_index=ignore_index
        )
        losses = losses.reshape(batch_size, maxl)
        loss = torch.mean(torch.sum(losses, dim=-1))
        return loss

    def synthon_forward(self, prod_graph):
        prod_n_emb, prod_e_emb = self.GNN(prod_graph)
        AC_logits = self.Cchange(prod_n_emb).squeeze(dim=-1)
        prod_e_logits = self.syn_e_pred(prod_e_emb)

        return AC_logits, prod_e_logits, prod_n_emb, prod_e_emb


class SIM(torch.nn.Module):
    def __init__(self, q_dim, kv_dim, heads, dropout):
        super(SIM, self).__init__()
        self.Attn = DotMhAttn(
            Qdim=q_dim, emb_dim=q_dim, Odim=q_dim, Kdim=kv_dim,
            Vdim=kv_dim, num_heads=heads, dropout=dropout
        )

    def forward(self, x, other, key_padding_mask=None):
        attn_o, attn_w = self.Attn(
            query=x, key=other, value=other,
            key_padding_mask=key_padding_mask
        )
        return torch.relu(x + attn_o)
