from numpy import concatenate as npcat
import random
import torch
from sparse_backBone import GINBase, GATBase

from ogb.graphproppred.mol_encoder import AtomEncoder
from utils.chemistry_parse import(
    clear_map_number, get_synthon_edits, BOND_FLOAT_TO_IDX,
    cano_with_am, find_all_amap, remove_am_wo_cano
)
from utils.graph_utils import smiles2graph

from typing import Any, Dict, List, Tuple, Optional, Union
from torch_geometric.data import Data as GData
import math
import numpy as np
import multiprocessing
from tokenlizer import smi_tokenizer
from rdkit import Chem


class TransDataset(torch.utils.data.Dataset):
    def __init__(self, smiles, random_prob=0.0):
        super(TransDataset, self).__init__()

        self.smiles = smiles
        self.aug_prob = random_prob

    def __len__(self):
        return len(self.smiles)

    def randomize_smiles(self, smi):
        if 0 < self.aug_prob <= 1 and random.random() < self.aug_prob:
            if random.randint(0, 1) == 1:
                k = random.choice(self.smiles)
                return f'{smi}.{k}'
            else:
                mol = Chem.MolFromSmiles(smi)
                atms = [x.GetIdx() for x in mol.GetAtoms()]
                return Chem.MolToSmiles(
                    mol, rootedAtAtom=random.choice(atms), canonical=True
                )
        return smi

    def __getitem__(self, index):
        ret = ['<CLS>']
        out_smi = self.randomize_smiles(self.smiles[index])
        ret.extend(smi_tokenizer(out_smi))
        ret.append('<END>')

        return smiles2graph(out_smi, with_amap=False), ret


def col_fn_pretrain(data_batch):
    batch_size, max_node = len(data_batch), 0
    edge_idxes, edge_feats, node_feats, lstnode = [], [], [], 0
    batch, ptr, reats, node_per_graph = [], [0], [], []
    for idx, data in enumerate(data_batch):
        graph, ret = data
        num_nodes = graph['num_nodes']
        num_edges = graph['edge_index'].shape[1]
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

    all_batch_mask = torch.zeros((batch_size, max_node))
    for idx, mk in enumerate(node_per_graph):
        all_batch_mask[idx, :mk] = 1
    result['batch_mask'] = all_batch_mask.bool()

    return GData(**result), reats


class OnFlyDataset(torch.utils.data.Dataset):
    def __init__(
        self, prod_sm: List[str], reat_sm: List[str], aug_prob: float = 0,
    ):
        super(OnFlyDataset, self).__init__()
        self.prod_sm = prod_sm
        self.reat_sm = reat_sm

        self.aug_prob = aug_prob

    def __len__(self):
        return len(self.reat_sm)

    def process_prod(self, smi):
        if 0 < self.aug_prob <= 1 and random.random() < self.aug_prob:
            mol = Chem.MolFromSmiles(smi)
            atms = [x.GetIdx() for x in mol.GetAtoms()]
            return Chem.MolToSmiles(
                mol, rootedAtAtom=random.choice(atms), canonical=True
            )
        else:
            return cano_with_am(smi)

    def process_reac_via_prod(self, prod, reac):
        pro_atom_maps = find_all_amap(prod)
        reacts = reac.split('.')
        rea_atom_maps = [find_all_amap(x) for x in reacts]

        aligned_reactants = []
        for i, rea_map_num in enumerate(rea_atom_maps):
            for j, mapnum in enumerate(pro_atom_maps):
                if mapnum in rea_map_num:
                    mol = Chem.MolFromSmiles(reacts[i])
                    amap = {
                        x.GetAtomMapNum(): x.GetIdx() for x in mol.GetAtoms()
                    }

                    y_smi = Chem.MolToSmiles(
                        mol, rootedAtAtom=amap[mapnum], canonical=True
                    )

                    aligned_reactants.append((y_smi, j))
                    break

        aligned_reactants.sort(key=lambda x: x[1])
        return '.'.join(x[0] for x in aligned_reactants)

    def __getitem__(self, index):
        this_prod = self.process_prod(self.prod_sm[index])
        this_reac = self.process_reac_via_prod(this_prod, self.reat_sm[index])

        ret = ['<CLS>']
        ret.extend(smi_tokenizer(remove_am_wo_cano(this_reac)))
        ret.append('<END>')
        # print('[prod]', this_prod)
        # print('[reac]', this_reac)
        # print('[ret]', ret)

        Eatom, Hatom, Catom, deltaE, org_type = get_synthon_edits(
            this_reac, this_prod, consider_inner_bonds=True,
            return_org_type=True
        )

        graph, amap = smiles2graph(this_prod, with_amap=True)

        Ea = torch.zeros(graph['num_nodes']).long()
        Ha = torch.zeros(graph['num_nodes']).long()
        Ca = torch.zeros(graph['num_nodes']).long()

        Ea[[amap[t] for t in Eatom]] = 1
        Ha[[amap[t] for t in Hatom]] = 1
        Ca[[amap[t] for t in Catom]] = 1

        num_edges = graph['edge_feat'].shape[0]
        edge_label = torch.zeros(num_edges).long()

        new_type = {k: v[0] for k, v in org_type.items()}
        new_type.update({k: v[1] for k, v in deltaE.items()})
        new_edge = {}
        for (src, dst), ntype in new_type.items():
            src, dst = amap[src], amap[dst]
            ntype = BOND_FLOAT_TO_IDX[ntype]
            new_edge[(src, dst)] = new_edge[(dst, src)] = ntype

        for i in range(num_edges):
            src, dst = graph['edge_index'][:, i].tolist()
            edge_label[i] = new_edge[(src, dst)]

        return graph, Ea, Ha, Ca, edge_label, ret


def edit_col_fn(batch):
    Eatom, Hatom, Catom, reats = [], [], [], []
    batch_size, all_new, lstnode, lstedge = len(batch), [], 0, 0
    edge_idx, node_feat, edge_feat = [], [], []
    node_ptr, edge_ptr, node_batch, edge_batch = [0], [0], [], []
    max_node = max(x[0]['num_nodes'] for x in batch)
    batch_mask = torch.zeros(batch_size, max_node).bool()

    for idx, data in enumerate(batch):
        gp, Ea, Ha, Ca, elb, ret = data
        node_cnt, edge_cnt = gp['num_nodes'], gp['edge_index'].shape[1]

        node_feat.append(gp['node_feat'])
        edge_feat.append(gp['edge_feat'])
        edge_idx.append(gp['edge_index'] + lstnode)
        all_new.append(elb)
        Eatom.append(Ea)
        Hatom.append(Ha)
        Catom.append(Ca)
        reats.append(ret)

        batch_mask[idx, :node_cnt] = True

        lstnode += node_cnt
        lstedge += edge_cnt
        node_batch.append(np.ones(node_cnt, dtype=np.int64) * idx)
        edge_batch.append(np.ones(edge_cnt, dtype=np.int64) * idx)
        node_ptr.append(lstnode)
        edge_ptr.append(lstedge)

    result = {
        'x': torch.from_numpy(npcat(node_feat, axis=0)),
        "edge_attr": torch.from_numpy(npcat(edge_feat, axis=0)),
        'ptr': torch.LongTensor(node_ptr),
        'e_ptr': torch.LongTensor(edge_ptr),
        'batch': torch.from_numpy(npcat(node_batch, axis=0)),
        'e_batch': torch.from_numpy(npcat(edge_batch, axis=0)),
        'edge_index': torch.from_numpy(npcat(edge_idx, axis=-1)),
        'new_edge_types': torch.cat(all_new, dim=0),
        'EdgeChange': torch.cat(Eatom, dim=0),
        "ChargeChange": torch.cat(Catom, dim=0),
        "HChange": torch.cat(Hatom, dim=0),
        'num_nodes': lstnode,
        'num_edges': lstedge,
        'batch_mask': batch_mask
    }

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
        self.Hchange = torch.nn.Sequential(
            torch.nn.Linear(d_model, d_model),
            torch.nn.ReLU(),
            torch.nn.Linear(d_model, 2)
        )

        self.Cchange = torch.nn.Sequential(
            torch.nn.Linear(d_model, d_model),
            torch.nn.ReLU(),
            torch.nn.Linear(d_model, 2)
        )
        self.Echange = torch.nn.Sequential(
            torch.nn.Linear(d_model, d_model),
            torch.nn.ReLU(),
            torch.nn.Linear(d_model, 2)
        )

        self.edge_cls = torch.nn.Sequential(
            torch.nn.Linear(d_model, d_model),
            torch.nn.ReLU(),
            torch.nn.Linear(d_model, 5)
        )
        self.pos_enc = pos_enc
        self.output_layer = torch.nn.Sequential(
            torch.nn.Linear(d_model, d_model),
            torch.nn.ReLU(),
            torch.nn.Linear(d_model, token_size)
        )

    def graph2batch(
        self, node_feat: torch.Tensor, batch_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, max_node = batch_mask.shape
        answer = torch.zeros(batch_size, max_node, node_feat.shape[-1])
        answer = answer.to(node_feat.device)
        answer[batch_mask] = node_feat
        return answer

    def encode(self, graphs):
        node_feat, edge_feat = self.encoder(graphs)
        memory = self.graph2batch(node_feat, graphs.batch_mask)
        memory = self.pos_enc(memory)

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

    def forward(self, graphs, tgt, tgt_mask, tgt_pad_mask):
        tgt_emb = self.pos_enc(self.word_emb(tgt))
        node_feat, edge_feat = self.encoder(graphs)

        memory = self.graph2batch(node_feat, graphs.batch_mask)
        memory = self.pos_enc(memory)

        result = self.output_layer(self.decoder(
            tgt=tgt_emb, memory=memory, tgt_mask=tgt_mask,
            memory_key_padding_mask=torch.logical_not(graphs.batch_mask),
            tgt_key_padding_mask=tgt_pad_mask
        ))

        AH_logs = self.Hchange(node_feat)
        AE_logs = self.Echange(node_feat)
        AC_logs = self.Cchange(node_feat)

        edge_logs = self.edge_cls(edge_feat)

        return edge_logs, AH_logs, AE_logs, AC_logs, result


class PretrainModel(torch.nn.Module):
    def __init__(self, token_size, encoder, decoder, d_model, pos_enc):
        super(PretrainModel, self).__init__()
        self.word_emb = torch.nn.Embedding(token_size, d_model)
        self.encoder, self.decoder = encoder, decoder
        self.pos_enc = pos_enc
        self.output_layer = torch.nn.Sequential(
            torch.nn.Linear(d_model, d_model),
            torch.nn.ReLU(),
            torch.nn.Linear(d_model, token_size)
        )

    def graph2batch(
        self, node_feat: torch.Tensor, batch_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, max_node = batch_mask.shape
        answer = torch.zeros(batch_size, max_node, node_feat.shape[-1])
        answer = answer.to(node_feat)
        answer[batch_mask] = node_feat
        return answer

    def encode(self, graphs):
        node_feat, edge_feat = self.encoder(graphs)
        memory = self.graph2batch(node_feat, graphs.batch_mask)
        memory = self.pos_enc(memory)

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

    def forward(self, graphs, tgt, tgt_mask, tgt_pad_mask):

        memory, memory_pad = self.encode(graphs)
        result = self.decode(
            tgt=tgt, memory=memory, memory_padding_mask=memory_pad,
            tgt_padding_mask=tgt_pad_mask, tgt_mask=tgt_mask
        )

        return result
