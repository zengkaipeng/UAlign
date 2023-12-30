import random
import torch
from sparse_backBone import GINBase, GATBase

from ogb.graphproppred.mol_encoder import AtomEncoder
from utils.chemistry_parse import get_reaction_core, clear_map_number
from utils.graph_utils import smiles2graph

from typing import Any, Dict, List, Tuple, Optional, Union
from torch_geometric.data import Data as GData
import math
import numpy as np
import multiprocessing
from tokenlizer import smi_tokenizer


class OnFlyDataset(torch.utils.data.Dataset):
    def __init__(
        self, prod_sm: List[str], reat_sm: List[str], target: List[str],
        randomize: bool = False, aug_prob: float = 0, kekulize: bool = False
    ):
        super(OnFlyDataset, self).__init__()
        self.prod_sm = prod_sm
        self.reat_sm = reat_sm
        self.target = target

        self.randomize = randomize
        self.aug_prob = aug_prob
        self.kekulize = kekulize

    def __len__(self):
        return len(self.reat_sm)

    def process_reac(self, smi):
        if not self.randomize or not (0 < self.aug_prob < 1):
            return smi
        if random.random() < self.aug_prob:
            x = smi.split('.')
            random.shuffle(x)
            return '.'.join(x)
        else:
            return smi

    def __getitem__(self, index):
        ret = ['<CLS>']
        ret += smi_tokenizer(self.process_reac(self.target[index]))
        ret.append('<END>')

        x, y = get_reaction_core(
            r=self.reat_sm[index], p=self.prod_sm[index],
            kekulize=self.kekulize
        )

        graph, amap = smiles2graph(
            smiles_string=self.prod_sm[index],
            with_amap=True, kekulize=self.kekulize
        )

        node_label = torch.zeros(graph['num_nodes']).long()
        node_label[[amap[t] for t in x]] = 1

        num_edges = graph['edge_feat'].shape[0]

        es, edge_label = [], torch.zeros(num_edges).long()
        for edgs in y:
            src, dst, _, _ = edgs.split(':')
            src, dst = int(src), int(dst)
            if dst == 0:
                continue
            es.append((amap[src], amap[dst]))

        for idx, t in enumerate(graph['edge_index'][0]):
            src, dst = t.item(), graph['edge_index'][1][idx]
            if (src, dst) in es or (dst, src) in es:
                edge_label[idx] = 1

        return graph, node_label, edge_label, ret


def col_fn_unloop(data_batch):
    batch_size, node_label, max_node = len(data_batch), [], 0
    edge_idxes, edge_feats, node_feats, lstnode = [], [], [], 0
    edge_label, batch, ptr, reats = [], [], [0], []
    node_per_graph = []
    for idx, data in enumerate(data_batch):
        graph, n_lb, e_lb, ret = data
        num_nodes = graph['num_nodes']
        num_edges = graph['edge_index'].shape[1]

        node_label.append(n_lb)
        edge_label.append(e_lb)
        reats.append(ret)

        edge_idxes.append(graph['edge_index'] + lstnode)
        edge_feats.append(graph['edge_feat'])
        node_feats.append(graph['node_feat'])

        lstnode += num_nodes
        max_node = max(max_node, num_nodes)
        node_per_graph.append(num_nodes)
        batch.append(np.ones(num_nodes, dtype=np.int64) * idx)
        ptr.append(lstnode)

    result = {
        'edge_index': np.concatenate(edge_idxes, axis=-1),
        'edge_attr': np.concatenate(edge_feats, axis=0),
        'batch': np.concatenate(batch, axis=0),
        'x': np.concatenate(node_feats, axis=0),
        'ptr': np.array(ptr, dtype=np.int64)
    }

    result = {k: torch.from_numpy(v) for k, v in result.items()}
    result['num_nodes'] = lstnode
    result['node_label'] = torch.cat(node_label, dim=0)
    result['edge_label'] = torch.cat(edge_label, dim=0)

    all_batch_mask = torch.zeros((batch_size, max_node))
    for idx, mk in enumerate(node_per_graph):
        all_batch_mask[idx, :mk] = 1
    result['batch_mask'] = all_batch_mask.bool()

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

    def graph2batch(
        self, node_feat: torch.Tensor, batch_mask: torch.Tensor,
        batch_size: int, max_node: int
    ) -> torch.Tensor:
        answer = torch.zeros(batch_size, max_node, node_feat.shape[-1])
        answer = answer.to(node_feat.device)
        answer[batch_mask] = node_feat
        return answer

    def encode(self, graphs):
        node_feat, edge_feat = self.encoder(graphs)

        # batch_size = graphs.batch.max().item() + 1
        # max_mem_len = graphs.node_per_graph.max().item()
        batch_size, max_mem_len = graphs.batch_mask.shape
        memory = self.graph2batch(
            node_feat=node_feat, batch_mask=graphs.batch_mask,
            batch_size=batch_size, max_node=max_mem_len
        )
        return memory, torch.logical_not(graphs.batch_mask)

    def decode(
        self, tgt, memory, memory_padding_mask=None,
        tgt_mask=None, tgt_padding_mask=None
    ):
        tgt_emb = self.pos_enc(self.word_emb(tgt))
        result = self.decoder(
            tgt=tgt_emb, memory=memory, tgt_mask=tgt_mask,
            memory_key_padding_mask=memory_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask
        )
        return self.output_layer(result)

    def forward(self, graphs, tgt, tgt_mask, tgt_pad_mask, pred_core=False):
        tgt_emb = self.pos_enc(self.word_emb(tgt))
        node_feat, edge_feat = self.encoder(graphs)

        batch_size, max_mem_len = graphs.batch_mask.shape
        memory = self.graph2batch(
            node_feat=node_feat, batch_mask=graphs.batch_mask,
            batch_size=batch_size, max_node=max_mem_len
        )

        result = self.output_layer(self.decoder(
            tgt=tgt_emb, memory=memory, tgt_mask=tgt_mask,
            memory_key_padding_mask=torch.logical_not(graphs.batch_mask),
            tgt_key_padding_mask=tgt_pad_mask
        ))

        if pred_core:
            node_res = self.node_cls(node_feat)
            edge_res = self.edge_cls(edge_feat)
            return result, node_res, edge_res
        else:
            return result


class Acc_fn(torch.nn.Module):
    def __init__(self, ignore_index=-1):
        super(Acc_fn, self).__init__()
        self.ignore_index = ignore_index

    def forward(self, tgt, gt):
        tgt = tgt.argmax(dim=-1)
        mask = (gt != self.ignore_index)
        accs = (tgt[mask] == gt[mask]).sum()
        tots = mask.sum()
        return accs.item(), tots.item()
