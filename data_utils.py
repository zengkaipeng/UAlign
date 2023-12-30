import pandas
import os
from utils.chemistry_parse import (
    get_reaction_core, get_bond_info, BOND_FLOAT_TO_TYPE,
    BOND_FLOAT_TO_IDX, clear_map_number
)
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


def correct_trans_output(trans_pred, end_idx, pad_idx):
    batch_size, max_len = trans_pred.shape
    device = trans_pred.device
    x_range = torch.arange(0, max_len, 1).unsqueeze(0)
    x_range = x_range.repeat(batch_size, 1).to(device)

    y_cand = (torch.ones_like(trans_pred).long() * max_len + 12).to(device)
    y_cand[trans_pred == end_idx] = x_range[trans_pred == end_idx]
    min_result = torch.min(y_cand, dim=-1, keepdim=True)
    end_pos = min_result.values
    trans_pred[x_range > end_pos] = pad_idx
    return trans_pred


def eval_trans(trans_pred, trans_lb, return_tensor=False):
    batch_size, max_len = trans_pred.shape
    line_acc = torch.sum(trans_pred == trans_lb, dim=-1) == max_len
    line_acc = line_acc.cpu()
    return line_acc if return_tensor else (line_acc.sum().item(), batch_size)
