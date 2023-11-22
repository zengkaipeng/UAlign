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
        use_sim=False, pre_graph=True, heads=1, dropout=0.0, maxlen=2000
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
        self.use_sim, self.pre_graph = use_sim, pre_graph
        if self.use_sim:
            self.SIM_G = SIM(node_dim, node_dim, heads, dropout)
            self.SIM_L = SIM(node_dim, node_dim, heads, dropout)

        self.token_embeddings = torch.nn.Embedding(num_token, node_dim)
        self.PE = PositionalEncoding(node_dim, dropout, maxlen)
        self.trans_pred = torch.nn.Linear(node_dim, num_token)

    def add_type_emb(self, x, is_reac):
        batch_size, max_len = x.shape[:2]
        if is_reac:
            type_emb = self.reac_embedding.repeat(batch_size, max_len, 1)
        else:
            type_emb = self.prod_embedding.repeat(batch_size, max_len, 1)

        return self.emb_trans(torch.cat([x, type_emb], dim=-1)) + x

    def trans_enc_forward(self, word_emb, word_pad, graph_emb, graph_pad):
        word_emb = self.add_type_emb(word_emb, is_reac=True)
        word_emb = self.PE(word_emb)
        graph_emb = self.add_type_emb(graph_emb, is_reac=False)

        if self.pre_graph:
            trans_input = torch.cat([word_emb, graph_emb], dim=1)
            memory_pad = torch.cat([word_pad, graph_pad], dim=1)
            memory = self.trans_enc(trans_input, key_padding_mask=memory_pad)
        else:
            memory = self.trans_enc(word_emb, key_padding_mask=word_pad)
            memory = torch.cat([memory, graph_pad], dim=1)
            memory_pad = torch.cat([word_pad, graph_pad], dim=1)
        return memory, memory_pad

    def conn_forward(self, lg_emb, graph_emb, conn_edges, node_mask):
        useful_edges_mask = node_mask[conn_edges[:, 1]]
        useful_src, useful_dst = conn_edges[useful_edges_mask]
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
        trans_op, pad_idx=None, trans_ip_key_padding=None,
        trans_op_key_padding=None, trans_op_mask=None,
        trans_label=None, conn_label=None, mode='train'
    ):
        prod_n_emb, prod_e_emb = self.GNN(prod_graph)
        lg_n_emb, lg_e_emb = self.GNN(lg_graph)

        prod_n_logits = self.syn_n_pred(prod_n_emb)
        prod_e_logits = self.syn_e_pred(prod_e_emb)

        trans_ip = self.token_embeddings(trans_ip)
        trans_op = self.token_embeddings(trans_op)

        batched_prod_emb, prod_padding_mask = \
            make_memory_from_feat(prod_n_emb, prod_graph.batch_mask)
        memory, memory_pad = self.trans_enc_forward(
            trans_ip, trans_ip_key_padding,
            batched_prod_emb, prod_padding_mask
        )

        trans_pred = self.trans_pred(self.trans_dec(
            tgt=trans_op, memory=memory, tgt_mask=trans_op_mask,
            memory_key_padding_mask=prod_padding_mask,
            tgt_key_padding_mask=trans_op_key_padding
        ))

        lg_act_logits = self.lg_activate(lg_n_emb)
        if mode == 'train':
            lg_useful = (lg_graph.node_label > 0) | (lg_act_logits > 0)
        else:
            lg_useful = (lg_act_logits > 0)

        if self.use_sim:
            n_prod_emb, n_lg_emb = self.update_via_sim(
                prod_n_emb, prod_graph.batch_mask,
                lg_n_emb, lg_graph.batch_mask
            )
        else:
            n_prod_emb, n_lg_emb = prod_n_emb, lg_n_emb
        conn_logits, conn_mask = self.conn_forward(
            n_prod_emb, n_lg_emb, conn_edges, lg_useful
        )

        if mode == 'train':
            return self.loss_calc(
                prod_n_log=prod_n_logits,
                prod_e_log=prod_e_logits,
                prod_n_label=prod_graph.node_label,
                prod_e_label=prod_graph.edge_label,
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
        else:
            return prod_n_logits, prod_e_logits, lg_act_logits,\
                conn_logits, conn_mask, trans_pred

    def loss_calc(
        self, prod_n_log, prod_e_log, prod_n_label, prod_e_label,
        prod_n_batch, prod_e_batch, lg_n_log, lg_n_label, lg_n_batch,
        conn_lg, conn_lb, conn_batch, trans_pred, trans_lb, pad_idx
    ):
        syn_node_loss = self.scatter_loss_by_batch(
            prod_n_log, prod_n_label, prod_n_batch, cross_entropy
        )
        syn_edge_loss = self.scatter_loss_by_batch(
            prod_e_log, prod_e_label, prod_e_batch, cross_entropy
        )

        lg_act_loss = self.scatter_loss_by_batch(
            lg_n_log, lg_n_label, lg_n_batch,
            binary_cross_entropy_with_logits
        )

        conn_loss = self.scatter_loss_by_batch(
            conn_lg, conn_lb, conn_batch,
            binary_cross_entropy_with_logits
        )

        trans_loss = self.calc_trans_loss(trans_pred, trans_lb, pad_idx)
        return syn_node_loss, syn_edge_loss, lg_act_loss, conn_loss, trans_loss

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


class SIM(torch.nn.Module):
    def __init__(self, q_dim, kv_dim, heads, dropout):
        super(SIM, self).__init__()
        self.Attn = torch.nn.MultiheadAttention(
            embed_dim=q_dim, kdim=kv_dim, vdim=kv_dim,
            num_heads=heads, dropout=dropout, batch_first=True
        )

    def forward(self, x, other, key_padding_mask=None):
        return x + self.Attn(
            query=x, key=other, value=other,
            key_padding_mask=key_padding_mask
        )


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
