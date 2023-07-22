import pandas
import os
from utils.chemistry_parse import (
    get_reaction_core, get_bond_info, BOND_FLOAT_TO_TYPE,
    BOND_FLOAT_TO_IDX
)
from model import EditDataset
from utils.graph_utils import smiles2graph
from utils.chemistry_parse import clear_map_number
import random
import numpy as np
import torch
from tqdm import tqdm
import rdkit
from rdkit import Chem
import multiprocessing


def process_rxn_batch(
    reacts, prods, Q, rxn_class=None, kekulize=False,
    display_step=50000, display_fmt='{} / {} DONE'
):
    graphs, nodes, edge_types, ret = [], [], [], []
    for idx, prod in enumerate(prods):
        ret.append(clear_map_number(reacts[idx]))
        x, y = get_reaction_core(reacts[idx], prod, kekulize=kekulize)
        graph, amap = smiles2graph(prod, with_amap=True, kekulize=kekulize)
        graphs.append(graph)
        nodes.append([amap[t] for t in x])
        es = []
        for edgs in y:
            src, dst, _, _ = edgs.split(':')
            src, dst = int(src), int(dst)
            if dst == 0:
                continue
            es.append((amap[src], amap[dst]))
        edge_types.append(es)

        if (idx + 1) % display_step == 0:
            print(display_fmt.format(idx + 1, len(prods)))

    Q.put([graphs, nodes, edge_types, ret, rxn_class])


def create_sparse_dataset_mp(
    reacts, prods, num_proc, rxn_class=None, kekulize=False,
    randomize=False,  aug_prob=0, display_fmt='{} / {} DONE',
    display_step=50000, batch_size=200000
):
    graphs, nodes, edge_types, ret = [], [], [], []
    pool = multiprocessing.Pool(processes=num_proc)
    MpQ = multiprocessing.Manager().Queue()
    ppargs, all_len = [], len(prods)
    for idx in range(0, all_len, batch_size):
        ed_idx = idx + batch_size
        ppargs.append((
            reacts[idx: ed_idx], prods[idx: ed_idx], MpQ,
            None if rxn_class is None else rxn_class[idx: ed_idx],
            kekulize, display_step, display_fmt
        ))

    pool.starmap(process_rxn_batch, ppargs)
    pool.close()
    pool.join()

    rcls = None if rxn_class is None else []

    while not MpQ.empty():
        a, b, c, d, e = MpQ.get()
        graphs.extend(a)
        nodes.extend(b)
        edge_types.extend(c)
        ret.extend(d)
        if rcls is not None:
            rcls.extend(e)

    dataset = EditDataset(
        graphs, nodes, edge_types, ret, rcls,
        randomize=randomize, aug_prob=aug_prob
    )

    print('dataset_len', len(dataset))
    return dataset


def create_sparse_dataset(
    reacts, prods, rxn_class=None, kekulize=False,
    verbose=True, randomize=False, aug_prob=0
):
    graphs, nodes, edge_types, ret = [], [], [], []
    for idx, prod in enumerate(tqdm(prods, ascii=True) if verbose else prods):
        ret.append(clear_map_number(reacts[idx]))
        x, y = get_reaction_core(reacts[idx], prod, kekulize=kekulize)
        graph, amap = smiles2graph(prod, with_amap=True, kekulize=kekulize)
        graphs.append(graph)
        nodes.append([amap[t] for t in x])
        es = []
        for edgs in y:
            src, dst, _, _ = edgs.split(':')
            src, dst = int(src), int(dst)
            if dst == 0:
                continue
            es.append((amap[src], amap[dst]))
        edge_types.append(es)

    dataset = EditDataset(
        graphs, nodes, edge_types, ret, rxn_class,
        randomize=randomize, aug_prob=aug_prob
    )
    return dataset


def load_data(data_dir, part=None):
    if part is not None:
        df_train = pandas.read_csv(
            os.path.join(data_dir, f'canonicalized_raw_{part}.csv')
        )
    else:
        df_train = pandas.read_csv(data_dir)
    rxn_class, reacts, prods = [], [], []
    for idx, resu in enumerate(df_train['reactants>reagents>production']):
        rxn_class.append(df_train['class'][idx])
        rea, prd = resu.strip().split('>>')
        reacts.append(rea)
        prods.append(prd)
    return reacts, prods, rxn_class


def fix_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)


def generate_square_subsequent_mask(sz, device='cpu'):
    mask = (torch.triu(torch.ones((sz, sz))) == 1).transpose(0, 1)
    # mask = mask.float().masked_fill(mask == 0, float('-inf'))
    # mask = mask.masked_fill(mask == 1, float(0.0)).to(device)
    mask = (mask == 0).to(device)
    return mask


def generate_tgt_mask(tgt, tokenizer, pad='<PAD>', device='cpu'):
    PAD_IDX, siz = tokenizer.token2idx[pad], tgt.shape[1]
    tgt_pad_mask = (tgt == PAD_IDX).to(device)
    tgt_sub_mask = generate_square_subsequent_mask(siz, device)
    return tgt_pad_mask, tgt_sub_mask


if __name__ == '__main__':
    pass
