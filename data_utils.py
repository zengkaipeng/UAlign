import pandas
import os
from utils.chemistry_parse import (
    get_reaction_core, get_bond_info, BOND_FLOAT_TO_TYPE,
    BOND_FLOAT_TO_IDX, clear_map_number
)
from model import EditDataset
from utils.graph_utils import smiles2graph
import random
import numpy as np
import torch
from tqdm import tqdm
import rdkit
from rdkit import Chem
import multiprocessing

def load_ext_data(data_dir, part=None):
    if part is not None:
        fname = os.path.join(data_dir, f'extend_{part}.csv')
        df_train = pandas.read_csv(fname)
    else:
        df_train = pandas.read_csv(data_dir)
    rxn_class, reacts, prods, target = [], [], [], []
    for idx, resu in enumerate(df_train['reactants>reagents>production']):
        rxn_class.append(df_train['class'][idx])
        rea, prd = resu.strip().split('>>')
        reacts.append(rea)
        prods.append(prd)
        target.append(df_train['clean_reactant'][idx])
    return reacts, prods, rxn_class, target


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
