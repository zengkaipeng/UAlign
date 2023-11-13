import pandas
import os
from utils.chemistry_parse import (
    get_synthons, ACHANGE_TO_IDX, break_fragements
)
from utils.graph_utils import smiles2graph
from Dataset import OverallDataset, InferenceDataset
from Dataset import SynthonDataset
import random
import torch
import numpy as np
from tqdm import tqdm
from typing import Any, Dict, List, Tuple, Optional, Union


def create_edit_dataset(
    reacts: List[str], prods: List[str], rxn_class: Optional[List[int]] = None,
    kekulize: bool = False, verbose: bool = True,
):
    graphs, nodes, edges = [], [], []
    for idx, prod in enumerate(tqdm(prods) if verbose else prods):
        graph, amap = smiles2graph(prod, with_amap=True, kekulize=kekulize)
        graphs.append(graph)

        deltaH, deltaE = get_synthons(prod, reacs[idx], kekulize=kekulize)

        nodes.append({amap[k]: ACHANGE_TO_IDX[v] for k, v in deltaH.items()})
        this_edge = {}
        for (src, dst), (otype, ntype) in deltaE.items():
            src, dst = amap[src], amap[dst]
            if otype == ntype:
                this_edge[(src, dst)] = this_edge[(dst, src)] = 1
            elif ntype == 0:
                this_edge[(src, dst)] = this_edge[(dst, src)] = 2
            else:
                this_edge[(src, dst)] = this_edge[(dst, src)] = 0
        edges.append(this_edge)

    return SynthonDataset(graphs, nodes, edges, rxn_class=rxn_class)


def create_overall_dataset(
    reacts, prods, rxn_class=None, kekulize=False,
    verbose=True,
):
    graphs, nodes, edges, inv_amap = [], [], [], []
    for idx, prod in enumerate(tqdm(prods) if verbose else prods):
        graph, amap = smiles2graph(prod, with_amap=True, kekulize=kekulize)
        graphs.append(graph)

        deltaH, deltaE = get_synthons(prod, reacs[idx], kekulize=kekulize)

        nodes.append({amap[k]: ACHANGE_TO_IDX[v] for k, v in deltaH.items()})
        this_edge, break_edges = {}, set()
        for (src, dst), (otype, ntype) in deltaE.items():
            os, od = src, dst
            src, dst = amap[src], amap[dst]
            if otype == ntype:
                this_edge[(src, dst)] = this_edge[(dst, src)] = 1
            elif ntype == 0:
                this_edge[(src, dst)] = this_edge[(dst, src)] = 2
                break_edges.update([(od, os), (os, od)])
            else:
                this_edge[(src, dst)] = this_edge[(dst, src)] = 0

        edges.append(this_edge)

        synthon_str = break_fragements(prod, break_edges, canonicalize=False)
        



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


def eval_by_graph(
    node_pred, edge_pred, node_label, edge_label,
    node_batch, edge_batch
):
    node_acc, break_acc, break_cover = 0, 0, 0
    batch_size = node_batch.max().item() + 1
    for i in range(batch_size):
        this_node_mask = node_batch == i
        this_edge_mask = edge_batch == i

        this_nlb = node_label[this_node_mask]
        this_npd = node_pred[this_node_mask]
        node_acc += torch.all(this_nlb == this_npd).item()

        this_elb = edge_label[this_edge_mask]
        this_epd = edge_pred[this_edge_mask]

        p_break = this_epd == 2
        g_break = this_elb == 2
        bf = torch.all(p_break == g_break).item()

        p_break = this_epd == 0
        g_break = this_elb == 0
        inters = torch.logical_and(p_break, g_break)
        cf = torch.all(p_break == g_break).item()
        cc = torch.all(inters == g_break).item()

        break_acc += (bf & cf)
        break_cover += (bf & cc)

    return node_acc, break_acc, break_cover, batch_size


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


# def predict_synthon(batch_size, n_pred, e_pred, graph, n_types, e_types):
#     answer_n, answer_e = [], []
#     for idx in range(batch_size):
#         this_n_mask = graph.batch == idx
#         this_e_mask = graph.e_batch == idx
#         num_nodes = n_pred[this_n_mask].shape[0]
#         answer_n.append({ex: n_types[idx][ex] for ex in range(num_nodes)})
#         edge_res = {}
#         this_edge = graph.edge_index[this_e_mask].T
#         this_epd = e_pred[this_e_mask]
#         for edx, res in enumerate(this_epd):
#             if res.item() == 1:
#                 continue
#             row, col = this_edge[:, edx].tolist()
#             row -= graph.ptr[idx].item()
#             col -= graph.ptr[idx].item()
#             if (row, col) not in edge_res and (col, row not in edge_res):
#                 edge_res[(row, col)] = e_types[idx][(row, col)]
#         answer_e.append(edge_res)
#     return answer_n, answer_e

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
