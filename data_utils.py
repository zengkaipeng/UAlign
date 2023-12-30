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


def load_data(data_dir, part):
    df_train = pandas.read_csv(
        os.path.join(data_dir, f'canonicalized_raw_{part}.csv')
    )
    rxn_class, reacts, prods = [], [], []
    for idx, resu in enumerate(df_train['reactants>reagents>production']):
        rxn_class.append(df_train['class'][idx])
        rea, prd = resu.strip().split('>>')
        reacts.append(rea)
        prods.append(prd)
    return reacts, prods, rxn_class


def eval_by_batch(pred, label, batch, return_tensor=False, batch_size=None):
    if batch_size is None:
        batch_size = batch.max().item() + 1

    accs = torch.zeros(batch_size).bool()

    for i in range(batch_size):
        this_mask = batch == i

        if torch.any(this_mask):
            this_lb = label[this_mask]
            this_pd = pred[this_mask]
            accs[i] = torch.all(this_lb == this_pd).item()
        else:
            accs[i] = True

    if not return_tensor:
        return accs.sum().item(), batch_size
    else:
        return accs


def convert_edge_log_into_labels(
    logits, edge_index, mod='sigmoid', return_dict=False
):
    def covert_into_dict(edge_logs, mode):
        result = {}
        if mode == 'sigmoid':
            for (row, col), v in edge_logs.items():
                if (row, col) not in result and (col, row) not in result:
                    result[(row, col)] = 1 if v >= 0.5 else 0
        else:
            for (row, col), v in edge_logs.items():
                if (row, col) not in result and (col, row) not in result:
                    result[(row, col)] = v.argmax().item()
        return result

    def convert_into_tensor(edge_logs, edge_index, mode):
        num_edges = edge_index.shape[1]
        result = torch.zeros(num_edges).long().to(edge_index.device)
        if mode == 'sigmoid':
            for idx, (row, col) in enumerate(edge_index.T):
                row, col = row.item(), col.item()
                result[idx] = 1 if edge_logs[(row, col)] >= 0.5 else 0
        else:
            for idx, (row, col) in enumerate(edge_index.T):
                row, col = row.item(), col.item()
                result[idx] = edge_logs[(row, col)].argmax().item()
        return result

    if mod == 'sigmoid':
        edge_logs = {}
        for idx, p in enumerate(logits):
            row, col = edge_index[:, idx]
            row, col = row.item(), col.item()
            p = p.sigmoid().item()
            if (row, col) not in edge_logs:
                edge_logs[(row, col)] = edge_logs[(col, row)] = p
            else:
                real_log = (edge_logs[(row, col)] + p) / 2
                edge_logs[(row, col)] = edge_logs[(col, row)] = real_log

        edge_pred = covert_into_dict(edge_logs, mode=mod) if return_dict\
            else convert_into_tensor(edge_logs, edge_index, mode=mod)

    else:
        edge_logs = {}
        logits = torch.softmax(logits, dim=-1)
        for idx, p in enumerate(logits):
            row, col = edge_index[:, idx]
            row, col = row.item(), col.item()
            if (row, col) not in edge_logs:
                edge_logs[(row, col)] = edge_logs[(col, row)] = p
            else:
                real_log = (edge_logs[(row, col)] + p) / 2
                edge_logs[(row, col)] = edge_logs[(col, row)] = real_log

        edge_pred = covert_into_dict(edge_logs, mode=mod) if return_dict\
            else convert_into_tensor(edge_logs, edge_index, mode=mod)

    return edge_pred


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


def check_early_stop(*args):
    answer = True
    for x in args:
        answer &= all(t <= x[0] for t in x[1:])
    return answer


def convert_log_into_label(logits, mod='sigmoid'):
    if mod == 'sigmoid':
        pred = torch.zeros_like(logits)
        pred[logits >= 0] = 1
        pred[logits < 0] = 0
    elif mod == 'softmax':
        pred = torch.argmax(logits, dim=-1)
    else:
        raise NotImplementedError(f'Invalid mode {mod}')
    return pred
