import torch
from tokenlizer import smi_tokenizer
from utils.graph_utils import smiles2graph
import numpy as np
from typing import Any, Dict, List, Tuple, Optional, Union
from torch_geometric.data import Data as GData
from utils.chemistry_parse import find_all_amap, remove_am_wo_cano
import random
from rdkit import Chem



class TransDataset(torch.utils.data.Dataset):
    def __init__(self, smiles, reacts, mode='train'):
        super(TransDataset, self).__init__()
        self.smiles = smiles
        self.reacts = reacts
        self.mode = mode
        self.offset = len(self.smiles)

        assert mode in ['train', 'eval'], f'Invalid mode {mode}'

    def __len__(self):
        return len(self.smiles) + len(self.reacts)

    def randomize_smiles(self, smi):
        if random.randint(0, 1) == 1:
            k = random.choice(self.smiles)
            return f'{smi}.{k}'
        else:
            mol = Chem.MolFromSmiles(smi)
            return Chem.MolToSmiles(mol, doRandom=True)

    def random_react(self, smi):
        y = []
        for x in smi.split('.'):
            mol = Chem.MolFromSmiles(smi)
            y.append(Chem.MolToSmiles(mol, doRandom=True))
        return '.'.join(y)

    def __getitem__(self, index):
        ret = ['<CLS>']

        if index < self.offset:
            out_smi = self.randomize_smiles(self.smiles[index]) \
                if self.mode == 'train' else self.smiles[index]
        else:
            out_smi = self.reacts[index - self.offset]
            if self.mode == 'train':
                out_smi = self.random_react(out_smi)

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


class RetroDataset(torch.utils.data.Dataset):
    def __init__(
        self, prod_sm: List[str], reat_sm: List[str],
        rxn_cls: Optional[List[int]] = None, aug_prob: float = 0,
    ):
        super(RetroDataset, self).__init__()
        self.prod_sm = prod_sm
        self.reat_sm = reat_sm
        self.rxn_cls = rxn_cls
        self.aug_prob = aug_prob

    def __len__(self):
        return len(self.reat_sm)

    def remap_reac_prod(self, reac, prod):
        if 0 < self.aug_prob <= 1 and random.random() < self.aug_prob:
            # randomize the smiles and remap the reaction according to the
            # randomized result, have to make sure the amap number of
            # prod is int the range of 1 -> num atoms
            mol = Chem.MolFromSmiles(prod)
            temp_x = Chem.MolToSmiles(mol, doRandom=True)
            all_ams = find_all_amap(temp_x)
            remap = {v: idx + 1 for idx, v in enumerate(all_ams)}

            r_mol = Chem.MolFromSmiles(reac)

            for x in mol.GetAtoms():
                old_num = x.GetAtomMapNum()
                x.SetAtomMapNum(remap.get(old_num, old_num))

            for x in r_mol.GetAtoms():
                old_num = x.GetAtomMapNum()
                x.SetAtomMapNum(remap.get(old_num, old_num))
            return Chem.MolToSmiles(r_mol), Chem.MolToSmiles(mol)
        else:
            # the data is canonicalized in data_proprecess
            return reac, prod

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
        this_reac, this_prod = self.remap_reac_prod(
            reac=self.reat_sm[index], prod=self.prod_sm[index]
        )
        this_reac = self.process_reac_via_prod(this_prod, this_reac)
        rxn = None if self.rxn_cls is None else self.rxn_cls[index]
        ret = ['<CLS>' if rxn is None else f'<RXN>_{rxn}']
        ret.extend(smi_tokenizer(remove_am_wo_cano(this_reac)))
        ret.append('<END>')

        graph = smiles2graph(this_prod, with_amap=False)

        return graph,  ret, rxn


def col_fn_retro(data_batch):
    batch_size, max_node = len(data_batch), 0
    edge_idxes, edge_feats, node_feats, lstnode = [], [], [], 0
    batch, ptr, reats, node_per_graph = [], [0], [], []
    node_rxn, edge_rxn = [], []
    for idx, data in enumerate(data_batch):
        graph, ret, rxn = data
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

        if rxn is not None:
            node_rxn.append(np.ones(num_nodes, dtype=np.int64) * rxn)
            edge_rxn.append(np.ones(num_edges, dtype=np.int64) * rxn)

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

    if len(node_rxn) > 0:
        node_rxn = np.concatenate(node_rxn, axis=0)
        edge_rxn = np.concatenate(edge_rxn, axis=0)
        result['node_rxn'] = torch.from_numpy(node_rxn)
        result['edge_rxn'] = torch.from_numpy(edge_rxn)

    return GData(**result), reats
