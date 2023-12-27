import pandas
import os
from utils.chemistry_parse import (
    ACHANGE_TO_IDX, break_fragements, get_all_amap, get_mol_belong,
    clear_map_number, BOND_FLOAT_TO_IDX, get_synthon_edits,
    get_leaving_group_synthon, edit_to_synthons
)
from utils.graph_utils import smiles2graph
from Dataset import OverallDataset, InferenceDataset
from Dataset import SynthonDataset
import random
import torch
import numpy as np
from tqdm import tqdm
from typing import Any, Dict, List, Tuple, Optional, Union
from itertools import permutations


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


def create_edit_dataset(
    reacts: List[str], prods: List[str], rxn_class: Optional[List[int]] = None,
    verbose: bool = True,
):
    graphs, n_edges = [], []
    Ea, Ha, Ca = [], [], []
    for idx, prod in enumerate(tqdm(prods) if verbose else prods):
        graph, amap = smiles2graph(prod, with_amap=True)
        graphs.append(graph)

        Eatom, Hatom, Catom, deltaEs, org_type = get_synthon_edits(
            reac=reacts[idx], prod=prod, consider_inner_bonds=False,
            return_org_type=True
        )
        new_type = {
            **{k: v[0] for k, v in org_type.items()},
            **{k: v[1] for k, v in deltaEs.items()}
        }

        new_edge = {}
        for (src, dst), ntype in new_type.items():
            src, dst = amap[src], amap[dst]
            ntype = BOND_FLOAT_TO_IDX[ntype]
            new_edge[(src, dst)] = new_edge[(dst, src)] = ntype
        n_edges.append(new_edge)
        Ea.append({amap[x] for x in Eatom})
        Ca.append({amap[x] for x in Catom})
        Ha.append({amap[x] for x in Hatom})

    return SynthonDataset(
        graphs=graphs, new_types=n_edges, Eatom=Ea, Hatom=Ha,
        Catom=Ca, rxn_class=rxn_class,
    )


def check_useful_synthons(synthons, belong):
    for syn in synthons:
        all_amap = get_all_amap(syn)
        this_belong = None
        for x in all_amap:
            if this_belong is None:
                this_belong = belong[x]
            elif this_belong != belong[x]:
                return False
    return True


def create_overall_dataset(
    reacts, prods, rxn_class=None, verbose=True,
    randomize=False, aug_prob=0, mode='train'
):
    graphs, nodes, n_edges, real_rxns = [], [], [], []
    lg_graphs, conn_cands, conn_labels = [], [], []
    trans_input, trans_output, lg_act = [], [], []
    Ea, Ha, Ca = [], [], []
    for idx, prod in enumerate(tqdm(prods) if verbose else prods):
        graph, amap = smiles2graph(prod, with_amap=True)

        Eatom, Hatom, Catom, deltaEs, org_type = get_synthon_edits(
            reac=reacts[idx], prod=prod, consider_inner_bonds=False,
            return_org_type=True
        )
        new_type = {
            **{k: v[0] for k, v in org_type.items()},
            **{k: v[1] for k, v in deltaEs.items()}
        }

        new_edge = {}
        for (src, dst), ntype in new_type.items():
            src, dst = amap[src], amap[dst]
            ntype = BOND_FLOAT_TO_IDX[ntype]
            new_edge[(src, dst)] = new_edge[(dst, src)] = ntype

        # n_edges.append(new_edge)
        # Ea.append({amap[x] for x in Eatom})
        # Ca.append({amap[x] for x in Catom})
        # Ha.append({amap[x] for x in Hatom})

        this_reac, belong = reacts[idx].split('.'), {}
        for tdx, reac in enumerate(this_reac):
            belong.update({k: tdx for k in get_all_amap(reac)})

        synthon_str = edit_to_synthons(
            prod, {k: v[1] for k, v in deltaEs.items()}
        )
        synthon_str = synthon_str.split('.')

        if len(synthon_str) != len(this_reac) or \
                not check_useful_synthons(synthon_str, belong):
            print('[INFO] synthons mismatch reactants')
            print(f'[SMI] {reacts[idx]}>>{prod}')
            continue

        lgs, _, conn_edgs = get_leaving_group_synthon(
            prod=prod, reac=reacts[idx], consider_inner_bonds=False
        )

        lg_graph, lg_amap = smiles2graph('.'.join(lgs), with_amap=True)

        syh_ips = [0] * len(this_reac)
        lg_ops = [[] for _ in range(len(this_reac))]

        for x in synthon_str:
            syh_ips[get_mol_belong(x, belong)] = x
        for x in lgs:
            lg_ops[get_mol_belong(x, belong)].append(x)

        lg_ops = ['.'.join(x) for x in lg_ops]

        this_cog, this_clb = [], []
        for tdx, x in enumerate(syh_ips):
            syn_amap_set = get_all_amap(x)
            lg_amap_set = get_all_amap(lg_ops[tdx])
            for a in syn_amap_set:
                for b in lg_amap_set:
                    this_cog.append((amap[a], lg_amap[b]))
                    this_etype = conn_edgs.get((a, b), 0)
                    this_clb.append(BOND_FLOAT_TO_IDX[this_etype])

        act_lbs = [0] * len(lg_amap)
        for a, b in conn_edgs.keys():
            act_lbs[lg_amap[b]] = 1

        syh_ips = [clear_map_number(x) for x in syh_ips]
        lg_ops = [clear_map_number(x) for x in lg_ops]

        assert mode in ['train', 'eval'], f'Invalid mode {mode}'

        if mode == 'train':
            idx_iter = permutations(range(len(this_reac)))
        else:
            idx_iter = list(range(len(this_reac)))
            idx_iter.sort(key=lambda x: len(syh_ips[x]))
            idx_iter = [idx_iter]

        for peru in idx_iter:
            t_input = '`'.join([syh_ips[x] for x in peru])
            t_output = '`'.join([lg_ops[x] for x in peru])

            # data adding encoder

            n_edges.append(new_edge)
            Ea.append({amap[x] for x in Eatom})
            Ca.append({amap[x] for x in Catom})
            Ha.append({amap[x] for x in Hatom})
            graphs.append(graph)

            # data adding lgs
            lg_graphs.append(lg_graph)
            lg_act.append(act_lbs)
            conn_cands.append(this_cog)
            conn_labels.append(this_clb)

            # trans
            trans_input.append(t_input)
            trans_output.append(t_output)

            if rxn_class is not None:
                real_rxns.append(rxn_class[idx])

    return OverallDataset(
        graphs=graphs, enc_edges=n_edges, lg_graphs=lg_graphs, Eatom=Ea,
        Hatom=Ha, Catom=Ca, lg_labels=lg_act, conn_edges=conn_cands,
        conn_labels=conn_labels, trans_input=trans_input,
        trans_output=trans_output, randomize=randomize, aug_prob=aug_prob,
        rxn_class=None if len(real_rxns) == 0 else real_rxns,
    )


def create_infernece_dataset(
    reacts, prods, rxn_class=None, kekulize=False,
    verbose=True,
):
    graphs, node_types, edge_types, smis = [], [], [], []
    for idx, prod in enumerate(tqdm(prods) if verbose else prods):
        # encoder_part
        encoder_graph, prod_amap = smiles2graph(
            prod, with_amap=True, kekulize=kekulize
        )
        graphs.append(encoder_graph)
        node_type = get_node_types(prods[idx])
        edge_type = get_edge_types(prods[idx], kekulize=kekulize)

        real_n_types = {prod_amap[k]: v for k, v in node_type.items()}
        real_e_types = {
            (prod_amap[x], prod_amap[y]): v
            for (x, y), v in edge_type.items()
        }
        node_types.append(real_n_types)
        edge_types.append(real_e_types)
        smis.append(clear_map_number(reacts[idx]))

    return InferenceDataset(
        reac_graph=graphs, prod_smiles=smis, rxn_class=rxn_class,
        reac_node_type=node_types, reac_edge_type=edge_types
    )


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


def fix_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)


def check_early_stop(*args):
    answer = True
    for x in args:
        answer &= all(t <= x[0] for t in x[1:])
    return answer


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


def avg_edge_logs(logits, edge_index, mod='sigmoid'):
    edge_logs = {}
    assert mod in ['softmax', 'sigmoid'], f'Invalid Mode {mod}'
    if mod == 'softmax':
        logits = torch.softmax(logits, dim=-1)
    else:
        logits = logits.sigmoid().tolist()
    for idx, p in enumerate(logits):
        row, col = edge_index[:, idx]
        row, col = row.item(), col.item()
        if (row, col) not in edge_logs:
            edge_logs[(row, col)] = edge_logs[(col, row)] = p
        else:
            real_log = (edge_logs[(row, col)] + p) / 2
            edge_logs[(row, col)] = edge_logs[(col, row)] = real_log

    if mod == 'softmax':
        edge_logs = {k: v.tolist() for k, v in edge_logs.items()}

    return {(a, b): v for (a, b), v in edge_logs.items() if a < b}


def seperate_encoder_graphs(G):
    batch_size = G.batch.max().item() + 1
    graphs, rxns = [], []
    for idx in range(batch_size):
        this_graph = {}
        this_node_mask = G.batch == idx
        this_edge_mask = G.e_batch == idx
        this_eidx = G.edge_index[:, this_edge_mask]
        this_eidx = (this_eidx - G.ptr[idx]).cpu().numpy()

        graphs.append({
            'node_feat': G.x[this_node_mask].cpu().numpy(),
            'edge_index': this_eidx, 'num_nodes': G.x[this_node_mask].shape[0],
            'edge_feat': G.edge_attr[this_edge_mask].cpu().numpy()
        })

        if G.get('node_rxn', None) is not None:
            rxns.append(G.node_rxn[G.ptr[idx]].item())
    return (graphs, rxns) if len(rxns) != 0 else graphs


def seperate_pred(pred, batch_size, batch):
    preds = []
    for idx in range(batch_size):
        this_mask = batch == idx
        preds.append(pred[this_mask])
    return preds


def seperate_dict(label_dict, num_nodes, batch, ptr):
    device = batch.device
    all_idx = torch.arange(num_nodes).to(device)
    batch2single = all_idx - ptr[batch]
    batch_size = batch.max().item() + 1
    e_labels = [{} for _ in range(batch_size)]

    for (row, col), v in label_dict.items():
        b_idx = batch[row].item()
        x = batch2single[row].item()
        y = batch2single[col].item()
        e_labels[b_idx][(x, y)] = v

    return e_labels


def filter_label_by_node(node_pred, edge_pred, edge_index):
    useful_node = node_pred > 0
    useful_edges = useful_node[edge_index[0]] & useful_node[edge_index[1]]
    edge_pred[~useful_edges] = 0
    return node_pred, edge_pred


def extend_label_by_edge(node_pred, edge_pred, edge_index):
    useful_node = torch.zeros_like(node_pred).bool()
    useful_edge = edge_pred > 0

    useful_node[edge_index[0, useful_edge]] = True
    useful_node[edge_index[1, useful_edge]] = True
    node_pred[useful_node] = 1
    return node_pred, edge_pred


def predict_synthon(n_pred, e_pred, graph, n_types, e_types):
    answer_n, answer_e = [], []
    for idx, this_n in enumerate(n_pred):
        this_e_idx = graph.edge_index[:, graph.e_batch == idx]
        num_nodes, offset = this_n.shape[0], graph.ptr[idx].item()
        answer_n.append({ex: n_types[idx][ex] for ex in range(num_nodes)})
        edge_res = {}
        for edx, res in enumerate(e_pred[idx]):
            if res.item() == 1:
                continue
            row, col = this_e_idx[:, edx].tolist()
            row, col = row - offset, col - offset
            if (row, col) not in edge_res and (col, row) not in edge_res:
                edge_res[(row, col)] = e_types[idx][(row, col)]
        answer_e.append(edge_res)
    return answer_n, answer_e


if __name__ == '__main__':
    print(load_data('../data/USPTO-50K', 'train'))
