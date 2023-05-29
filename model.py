import torch
from sparse_backBone import (
    GINBase, GATBase, SparseAtomEncoder, SparseBondEncoder
)

from typing import Any, Dict, List, Tuple, Optional, Union
from torch_geometric.data import Data as GData
import math
import numpy as np
from data_utils import generate_square_subsequent_mask


class EditDataset(torch.utils.data.Dataset):
    def __init__(
        self, graphs: List[Dict],
        activate_nodes: List[List],
        activate_edges: List[List],
        reat: List[str],
        rxn_class: Optional[List[int]] = None
    ):
        super(EditDataset, self).__init__()
        self.graphs = graphs
        self.activate_nodes = activate_nodes
        self.activate_edges = activate_edges
        self.rxn_class = rxn_class
        self.reat = reat

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, index):
        node_label = torch.zeros(self.graphs[index]['num_nodes']).long()
        node_label[self.activate_nodes[index]] = 1
        num_edges = self.graphs[index]['edge_feat'].shape[0]
        edge_label = torch.zeros(num_edges).long()
        edges = self.graphs[index]['edge_index']
        for idx, t in enumerate(edges[0]):
            src, dst = t.item(), edges[1][idx].item()
            if (src, dst) in self.activate_edges[index]:
                edge_label[idx] = 1
            if (dst, src) in self.activate_edges[index]:
                edge_label[idx] = 1

        if self.rxn_class is None:
            ret = ['<CLS>']
        else:
            ret = [f'<RXN_{self.rxn_class[index]}>']
        ret += list(self.reat[index])
        ret.append('<END>')

        if self.rxn_class is not None:
            return self.graphs[index], self.rxn_class[index], \
                node_label, edge_label, ret
        else:
            return self.graphs[index], node_label, edge_label, ret


def fc_collect_fn(data_batch):
    batch_size, rxn_class, node_label = len(data_batch), [], []
    edge_idxes, edge_feats, node_feats, lstnode = [], [], [], 0
    edge_label, batch, ptr = [], [], [0]
    edge_rxn, node_rxn, reats = [], [], []
    for idx, data in enumerate(data_batch):
        if len(data) == 4:
            graph, n_lb, e_lb, ret = data
        else:
            graph, r_class, n_lb, e_lb, ret = data
            rxn_class.append(r_class)
            node_rxn.append(np.ones(
                graph['num_nodes'], dtype=np.int64
            ) * r_class)
            edge_rxn.append(np.ones(
                graph['edge_feat'].shape[0], dtype=np.int64
            ) * r_class)

        node_label.append(n_lb)
        edge_label.append(e_lb)
        reats.append(ret)

        edge_idxes.append(graph['edge_index'] + lstnode)
        edge_feats.append(graph['edge_feat'])
        node_feats.append(graph['node_feat'])
        lstnode += graph['num_nodes']
        batch.append(np.ones(graph['num_nodes'], dtype=np.int64) * idx)
        ptr.append(lstnode)

    result = {
        'edge_index': np.concatenate(edge_idxes, axis=-1),
        'edge_attr': np.concatenate(edge_feats, axis=0),
        'batch': np.concatenate(batch, axis=0),
        'x': np.concatenate(node_feats, axis=0),
        'ptr': np.array(ptr, dtype=np.int64)
    }

    if len(rxn_class) != 0:
        result['node_rxn'] = np.concatenate(node_rxn, axis=0)
        result['edge_rxn'] = np.concatenate(edge_rxn, axis=0)
        result['rxn_class'] = np.array(rxn_class, dtype=np.int64)

    result = {k: torch.from_numpy(v) for k, v in result.items()}
    result['num_nodes'] = lstnode
    result['node_label'] = torch.cat(node_label, dim=0)
    result['edge_label'] = torch.cat(edge_label, dim=0)

    return GData(**result), reats


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


class Graph2Seq(torch.nn.Module):
    def __init__(self, token_size, encoder, decoder, d_model, pos_enc):
        super(Graph2Seq, self).__init__()
        self.word_emb = torch.nn.Embedding(token_size, d_model)
        self.encoder, self.decoder = encoder, decoder
        self.atom_encoder = SparseAtomEncoder(d_model)
        self.bond_encoder = SparseBondEncoder(d_model)
        self.node_cls = torch.nn.Sequential(
            torch.nn.Linear(d_model, d_model),
            torch.nn.ReLU(),
            torch.nn.Linear(d_model, 2)
        )
        self.edge_cls = torch.nn.Sequential(
            torch.nn.Linear(d_model, d_model),
            torch.nn.ReLU(),
            torch.nn.Linear(d_model, 2)
        )
        self.pos_enc = pos_enc
        self.output_layer = torch.nn.Sequential(
            torch.nn.Linear(d_model, d_model),
            torch.nn.ReLU(),
            torch.nn.Linear(d_model, token_size)
        )

    def batch_mask(
        self, ptr: torch.Tensor, max_node: int, batch_size: int
    ) -> torch.Tensor:
        num_nodes = ptr[1:] - ptr[:-1]
        mask = torch.arange(max_node).repeat(batch_size, 1)
        mask = mask.to(num_nodes.device)
        return mask < num_nodes.reshape(-1, 1)

    def graph2batch(
        self, node_feat: torch.Tensor, batch_mask: torch.Tensor,
        batch_size: int, max_node: int
    ) -> torch.Tensor:
        answer = torch.zeros(batch_size, max_node, node_feat.shape[-1])
        answer = answer.to(node_feat.device)
        answer[batch_mask] = node_feat
        return answer

    def max_node_ptr(self, ptr):
        num_nodes = ptr[1:] - ptr[:-1]
        return num_nodes.max()

    def encode(self, graphs):
        node_feat, edge_feat = self.encoder(
            node_feats=self.atom_encoder(graphs.x, rxn_class=n_rxn),
            edge_feats=self.bond_encoder(graphs.edge_attr, rxn_class=e_rxn),
            edge_index=graphs.edge_index
        )

        batch_size = len(graphs.ptr) - 1
        max_mem_len = self.max_node_ptr(graphs.ptr)
        batch_mask = self.batch_mask(graphs.ptr, max_mem_len, batch_size)
        memory = self.graph2batch(
            node_feat=node_feat, batch_mask=batch_mask,
            batch_size=batch_size, max_node=max_mem_len
        )
        return memory, torch.logical_not(batch_mask)

    def decode(
        self, memory, memory_padding_mask=None, tgt_mask=None,
        tgt_padding_mask=None, to_cls=True,
    ):
        tgt_emb = self.pos_enc(self.word_emb(tgt))
        result = self.decoder(
            tgt=tgt_emb, memory=memory, tgt_mask=tgt_mask,
            memory_key_padding_mask=memory_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask
        )
        return self.output_layer(result) if to_cls else result

    def forward(self, graphs, tgt, tgt_mask, tgt_pad_mask, pred_core=False):
        tgt_emb = self.pos_enc(self.word_emb(tgt))
        n_rxn = getattr(graphs, 'node_rxn', None)
        e_rxn = getattr(graphs, 'edge_rxn', None)
        node_feat, edge_feat = self.encoder(
            node_feats=self.atom_encoder(graphs.x, rxn_class=n_rxn),
            edge_feats=self.bond_encoder(graphs.edge_attr, rxn_class=e_rxn),
            edge_index=graphs.edge_index
        )

        batch_size = len(graphs.ptr) - 1
        max_mem_len = self.max_node_ptr(graphs.ptr)
        batch_mask = self.batch_mask(graphs.ptr, max_mem_len, batch_size)
        memory = self.graph2batch(
            node_feat=node_feat, batch_mask=batch_mask,
            batch_size=batch_size, max_node=max_mem_len
        )

        result = self.output_layer(self.decoder(
            tgt=tgt_emb, memory=memory, tgt_mask=tgt_mask,
            memory_key_padding_mask=torch.logical_not(batch_mask),
            tgt_key_padding_mask=tgt_pad_mask
        ))

        if pred_core:
            node_res = self.node_cls(node_feat)
            edge_res = self.edge_cls(edge_feat)
            return result, node_res, edge_res
        else:
            return result


def greedy_infernece_one(
    model, tokenizer, begin_token='<CLS>', end_token='<END>',
    graph, device, max_len, use_class=False, rxn_class=None
):
    model = model.eval()
    if use_class and rxn_class is None:
        raise NotImplementedError('require rxn_class information')
    tgt = torch.LongTensor([tokenizer.encode1d([begin_token])])
    tgt = tgt.to(device)

    end_id = tokenizer.token2idx[end_token]
    with torch.no_grad():
        memory, memory_pad_mask = model.encode(graph)
        for idx in range(max_len):
            tgt_mask = generate_square_subsequent_mask(tgt.shape[1])
            result = model.decode(
                memory=memory, tgt_mask=tgt_mask,
                memory_padding_mask=mem_pad_mask
            )
            result = result[:, -1].argmax(dim=-1)
            # [1, 1]
            if result.item() == end_token:
                break
            tgt = torch.cat([tgt, result], dim=-1)

    tgt_list = tgt.tolist()[0]
    answer = tokenizer.decode1d(tgt_list)
    answer = answer.replace(end_token, "").replace(begin_token, "")
    return answer
