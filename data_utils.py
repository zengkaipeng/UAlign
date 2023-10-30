import pandas
import os
from utils.chemistry_parse import (
    get_reaction_core, get_bond_info, BOND_FLOAT_TO_TYPE,
    BOND_FLOAT_TO_IDX, get_modified_atoms_bonds,
    get_node_types, get_edge_types, clear_map_number
)
from utils.graph_utils import smiles2graph
from Dataset import OverallDataset, InferenceDataset
from Dataset import BinaryEditDataset
import random
import torch
import numpy as np
from tqdm import tqdm
from typing import Any, Dict, List, Tuple, Optional, Union


def create_edit_dataset(
    reacts, prods, rxn_class=None, kekulize=False, verbose=True,
):
    graphs, nodes, edges = [], [], []
    for idx, prod in enumerate(tqdm(prods) if verbose else prods):
        x, y = get_modified_atoms_bonds(reacts[idx], prod, kekulize=kekulize)
        graph, amap = smiles2graph(prod, with_amap=True, kekulize=kekulize)
        graphs.append(graph)
        nodes.append([amap[t] for t in x])
        edges.append([(amap[i], amap[j]) for i, j in y])

    return BinaryEditDataset(graphs, nodes, edges, rxn_class=rxn_class)


def extend_amap(amap, node_list):
    result, max_node = {}, max(amap.values())
    for node in node_list:
        if node not in amap:
            result[node] = max_node + 1
            max_node += 1
        else:
            result[node] = amap[node]
    return result


def create_overall_dataset(
    reacts, prods, rxn_class=None, kekulize=False,
    verbose=True, pos_enc='none', **kwargs
):
    graphs, nodes, edges = [], [], []
    node_types, edge_types = [], []
    for idx, prod in enumerate(tqdm(prods) if verbose else prods):
        # encoder_part
        x, y = get_modified_atoms_bonds(reacts[idx], prod, kekulize)
        encoder_graph, prod_amap = smiles2graph(
            prod, with_amap=True, kekulize=kekulize
        )
        encoder_graph = add_pos_enc(encoder_graph, method=pos_enc, **kwargs)
        graphs.append(encoder_graph)
        nodes.append([prod_amap[t] for t in x])
        edges.append([(prod_amap[i], prod_amap[j]) for i, j in y])

        node_type = get_node_types(reacts[idx])
        extended_amap = extend_amap(prod_amap, node_type.keys())
        edge_type = get_edge_types(reacts[idx], kekulize=kekulize)

        real_n_types = {extended_amap[k]: v for k, v in node_type.items()}
        real_e_types = {
            (extended_amap[x], extended_amap[y]): v
            for (x, y), v in edge_type.items()
        }
        node_types.append(real_n_types)
        edge_types.append(real_e_types)
    return OverallDataset(
        graphs=graphs, activate_nodes=nodes, changed_edges=edges,
        decoder_node_type=node_types, decoder_edge_type=edge_types,
        rxn_class=rxn_class
    )


def create_infernece_dataset(
    reacts, prods, rxn_class=None, kekulize=False,
    verbose=True, pos_enc='none', **kwargs
):
    graphs, node_types, edge_types, smis = [], [], [], []
    for idx, prod in enumerate(tqdm(prods) if verbose else prods):
        # encoder_part
        encoder_graph, prod_amap = smiles2graph(
            prod, with_amap=True, kekulize=kekulize
        )
        encoder_graph = add_pos_enc(encoder_graph, method=pos_enc, **kwargs)
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
    node_batch, edge_batch,
):
    node_fit, node_cover, edge_fit, edge_cover = [0] * 4
    batch_size = node_batch.max().item() + 1
    for i in range(batch_size):
        this_node_mask = node_batch == i
        this_edge_mask = edge_batch == i

        this_elb = edge_label[this_edge_mask] > 0
        this_epd = edge_pred[this_edge_mask] > 0

        e_inters = torch.logical_and(this_elb, this_epd)
        ef = torch.all(this_elb == this_epd).item()
        ec = torch.all(this_elb == e_inters).item()

        edge_fit += ef
        edge_cover += ec

        this_nlb = node_label[this_node_mask] > 0
        this_npd = node_pred[this_node_mask] > 0

        inters = torch.logical_and(this_nlb, this_npd)
        nf = torch.all(this_nlb == this_npd).item()
        nc = torch.all(this_nlb == inters).item()

        node_fit += nf
        node_cover += nc
    return node_fit, node_cover, edge_fit, edge_cover


def eval_by_node(
    node_pred, edge_pred, node_label, edge_label,
    node_batch, edge_batch, edge_index
):
    cover, fit = 0, 0
    batch_size = node_batch.max().item() + 1
    for i in range(batch_size):
        this_node_mask = node_batch == i
        this_edge_mask = edge_batch == i
        this_edge_index = edge_index[:, this_edge_mask]
        this_src, this_dst = this_edge_index

        useful_mask = torch.logical_and(
            node_pred[this_src] > 0, node_pred[this_dst] > 0
        )
        this_nlb = node_label[this_node_mask] > 0
        this_npd = node_pred[this_node_mask] > 0

        inters = torch.logical_and(this_nlb, this_npd)
        nf = torch.all(this_nlb == this_npd).item()
        nc = torch.all(this_nlb == inters).item()

        if torch.any(useful_mask):
            this_elb = edge_label[this_edge_mask][useful_mask] > 0
            this_epd = edge_pred[this_edge_mask][useful_mask] > 0
            e_inters = torch.logical_and(this_elb, this_epd)
            ef = torch.all(this_elb == this_epd).item()
            ec = torch.all(this_elb == e_inters).item()
        else:
            ec = ef = True

        cover += (nc & ec)
        fit += (nf & ef)
    return cover, fit


def eval_by_edge(
    node_pred, edge_pred, node_label, edge_label,
    node_batch, edge_batch, edge_index, node_ptr
):
    cover, fit = 0, 0
    batch_size = node_batch.max().item() + 1
    for i in range(batch_size):
        this_node_mask = node_batch == i
        this_edge_mask = edge_batch == i
        this_edge_index = edge_index[:, this_edge_mask]
        this_src, this_dst = this_edge_index

        this_elb = edge_label[this_edge_mask] > 0
        this_epd = edge_pred[this_edge_mask] > 0

        e_inters = torch.logical_and(this_elb, this_epd)
        ef = torch.all(this_elb == this_epd).item()
        ec = torch.all(this_elb == e_inters).item()

        this_nlb = node_label[this_node_mask] > 0
        this_npd = node_pred[this_node_mask] > 0
        if torch.any(this_epd).item():
            this_npd[this_src[this_epd] - node_ptr[i]] = True
            this_npd[this_dst[this_epd] - node_ptr[i]] = True
        inters = torch.logical_and(this_nlb, this_npd)
        nf = torch.all(this_nlb == this_npd).item()
        nc = torch.all(this_nlb == inters).item()

        cover += (nc & ec)
        fit += (nf & ef)
    return cover, fit


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

        graphs.append({
            'node_feat': G.x[this_node_mask].cpu().numpy(),
            'edge_index': G.edge_index[:, this_edge_mask].cpu().numpy(),
            'edge_feat': G.edge_attr[this_edge_mask].cpu().numpy(),
            'num_nodes': G.x[this_node_mask].shape[0]
        })

        if G.get('node_rxn', None) is not None:
            rxns.append(G.node_rxn[G.ptr[idx]].item())
    return (graphs, rxns) if len(rxns) != 0 else graphs


def seperate_pred(node_pred, edge_pred, G):
    batch_size = G.batch.max().item() + 1
    node_preds, edge_preds = [], []
    for idx in range(batch_size):
        this_node_mask = G.batch == idx
        this_edge_mask = G.e_batch == idx
        node_preds.append(node_pred[this_node_mask])
        edge_preds.append(edge_pred[this_edge_mask])

    return node_preds, edge_preds


def seperate_dict(label_dict, G):
    all_idx = torch.arange(G.num_nodes).to(G.batch.device)
    batch2single = all_idx - G.ptr[G.batch]
    batch_size = G.batch.max().item() + 1

    e_labels = [{} for _ in range(batch_size)]

    for (row, col), v in label_dict.items():
        b_idx = G.batch[row].item()
        x = batch2single[row].item()
        y = batch2single[col].item()
        e_labels[b_idx][(x, y)] = v
    return e_labels


if __name__ == '__main__':
    print(load_data('../data/USPTO-50K', 'train'))
