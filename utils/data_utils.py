import json
import pickle
import pandas
import os
from utils.graph_utils import smiles2graph
from utils.chemistry_parse import clear_map_number
import random
import numpy as np
import torch
from tqdm import tqdm
import rdkit
from rdkit import Chem
import multiprocessing
from utils.tokenlizer import DEFAULT_SP, Tokenizer
import time


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


def load_moles(data_dir, part, verbose=False):
    df_train = pandas.read_csv(
        os.path.join(data_dir, f'canonicalized_raw_{part}.csv')
    )
    moles, reacts = set(), set()
    iterx = df_train['reactants>reagents>production']
    if verbose:
        iterx = tqdm(iterx)
    for resu in iterx:
        rea, prd = resu.strip().split('>>')
        rea = clear_map_number(rea)
        prd = clear_map_number(prd)
        moles.update(rea.split('.'))
        moles.update(prd.split('.'))
        if '.' in rea:
            reacts.add(rea)
        if '.' in prd:
            reacts.add(prd)
    return list(moles), list(reacts)


def create_log_model(base_log, log_name=''):
    timestamp = log_name if log_name != '' else str(time.time())
    if not os.path.exists(base_log):
        os.makedirs(base_log)
    detail_log_dir = os.path.join(base_log, f'log-{timestamp}.json')
    detail_model_dir = os.path.join(base_log, f'mod-{timestamp}.pth')
    token_path = os.path.join(base_log, f'token-{timestamp}.pkl')
    return detail_log_dir, detail_model_dir, token_path


def init_tokenizer(token_path='', checkpoint='', token_ckpt=''):
    if checkpoint != '':
        assert token_ckpt != '', \
            'require token_ckpt when checkpoint is given'
        with open(token_ckpt, 'rb') as fin:
            tokenizer = pickle.load(fin)
    else:
        assert token_path != '', 'file containing all tokens are required'
        sp_token = DEFAULT_SP | set([f"<RXN>_{i}" for i in range(11)])
        with open(token_path) as fin:
            tokenizer = Tokenizer(json.load(fin), sp_token)
    return tokenizer


def dump_tokenizer(tokenizer, token_dir):
    with open(token_dir, 'wb') as fout:
        pickle.dump(tokenizer, fout)


def fix_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)


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
