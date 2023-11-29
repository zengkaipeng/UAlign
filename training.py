from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
import torch
from utils.chemistry_parse import convert_res_into_smiles
from data_utils import (
    eval_by_graph,
    convert_log_into_label,
    convert_edge_log_into_labels
)
from data_utils import predict_synthon, generate_tgt_mask


def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):
    def f(x):
        if x >= warmup_iters:
            return 1
        alpha = float(x) / warmup_iters
        return warmup_factor * (1 - alpha) + alpha

    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


def train_sparse_edit(loader, model, optimizer, device, warmup=True):
    model = model.train()
    node_loss, edge_loss = [], []
    if warmup:
        warmup_iters = len(loader) - 1
        warmup_sher = warmup_lr_scheduler(optimizer, warmup_iters, 5e-2)
    for graph in tqdm(loader):
        graph = graph.to(device)
        _, _, loss_node, loss_edge = model(graph, ret_loss=True)

        optimizer.zero_grad()
        (loss_node + loss_edge).backward()
        optimizer.step()

        node_loss.append(loss_node.item())
        edge_loss.append(loss_edge.item())

        if warmup:
            warmup_sher.step()
    return np.mean(node_loss), np.mean(edge_loss)


def eval_sparse_edit(loader, model, device):
    model = model.eval()
    node_acc, break_acc, break_cover, tot = [0] * 4
    for graph in tqdm(loader):
        graph = graph.to(device)
        with torch.no_grad():
            node_logs, edge_logs = model(graph, ret_loss=False)
            node_pred = convert_log_into_label(node_logs, mod='softmax')
            edge_pred = convert_edge_log_into_labels(
                edge_logs, graph.edge_index,
                mod='softmax', return_dict=False
            )

        na, ba, bc, batch_size = eval_by_graph(
            node_pred, edge_pred, graph.node_label,
            graph.edge_label, graph.batch, graph.e_batch
        )
        node_acc += na
        break_acc += ba
        break_cover += bc
        tot += batch_size

    result = {
        'node_acc': node_acc / tot, 'break_acc': break_acc / tot,
        'break_cover': break_cover / tot,
    }
    return result


def train_overall(
    model, loader, optimizer, device, tokenizer
    pad_token, warmup=True,
):
    enc_nl, enc_el, lg_act, conn, tras = [[] for _ in range(5)]
    model = model.train()
    if warmup:
        warmup_iters = len(loader) - 1
        warmup_sher = warmup_lr_scheduler(optimizer, warmup_iters, 5e-2)

    pad_idx = tokenizer.token2idx[pad_token]
    for data in tqdm(loader):
        prod_graph, lg_graph, conn_es, conn_ls, conn_b, tips, tops, grxn = data
        prod_graph = prod_graph.to(device)
        lg_graph = lg_graph.to(device)
        conn_es = conn_es.to(device)
        conn_ls = conn_ls.to(device)
        conn_b = conn_b.to(device)

        tips = tokenizer.encode_2d(tips)
        tops = tokenizer.encode_2d(tops)

        trans_ip_mask = tips == pad_idx

        trans_dec_ip = tops[:, :-1]
        trans_dec_op = tops[:, 1:]

        trans_op_mask, diag_mask = generate_tgt_mask(
            trans_dec_ip, tokenizer, pad_token, device=)
        if grxn is not None:
            grxn = grxn.to(device)

        losses = model(
            prod_graph=prod_graph, lg_graph=lg_graph, trans_ip=tips,
            conn_edges=conn_es, conn_batch=conn_b, trans_op=trans_dec_ip,
            grapg_rxn=grxn, pad_idx=pad_idx,  trans_op_mask=diag_mask,
            trans_ip_key_padding=trans_ip_mask,
            trans_op_key_padding=trans_op_mask, trans_label=trans_dec_op,
            conn_label=conn_ls, mode='train'
        )

        syn_node_loss, syn_edge_loss, lg_act_loss, \
            conn_loss, trans_loss = losses

        loss = syn_node_loss + syn_edge_loss + lg_act_loss \
            + conn_loss + trans_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        enc_nl.append(syn_node_loss.item())
        enc_el.append(syn_edge_loss.item())
        lg_act.append(lg_act_loss.item())
        conn.append(conn_loss.item())
        tras.append(trans_loss.item())

        if warmup:
            warmup_sher.step()

    return {
        'enc_node_loss': np.mean(enc_nl), 'enc_edge_loss': np.mean(enc_el),
        'lg_act_loss': np.mean(lg_act), 'conn_loss': np.mean(conn_loss),
        'trans_loss': np.mean(tras)
    }


def eval_overall(
    model, loader, device, tokenizer, pad_toekn,
    end_token,
):
    model, eval_res = model.eval(), [], []
    loss_cur = {
        'syn_node_loss': [], 'syn_edge_loss': [], 'all': [],
        'lg_act_loss': [], 'conn_loss': [], 'trans_loss': []
    }
    metrics = {
        'synthon': {'node_acc': [], 'break_acc': [], 'break_cover': []},
        'lg': [], 'conn': {'cover': [], 'fit': []}, 'all': []
    }
    for data in tqdm(loader):
        prod_graph, lg_graph, conn_es, conn_ls, conn_b, tips, tops, grxn = data
        prod_graph = prod_graph.to(device)
        lg_graph = lg_graph.to(device)
        conn_es = conn_es.to(device)
        conn_ls = conn_ls.to(device)
        conn_b = conn_b.to(device)

        tips = tokenizer.encode_2d(tips)
        tops = tokenizer.encode_2d(tops)

        trans_ip_mask = tips == pad_idx
        trans_dec_ip = tops[:, :-1]
        trans_dec_op = tops[:, 1:]

        trans_op_mask, diag_mask = generate_tgt_mask(
            trans_dec_ip, tokenizer, pad_token, device=)
        if grxn is not None:
            grxn = grxn.to(device)

        with torch.no_grad():
            preds, losses = model(
                prod_graph=prod_graph, lg_graph=lg_graph, trans_ip=tips,
                conn_edges=conn_es, conn_batch=conn_b, trans_op=trans_dec_ip,
                grapg_rxn=grxn, pad_idx=pad_idx,  trans_op_mask=diag_mask,
                trans_ip_key_padding=trans_ip_mask,
                trans_op_key_padding=trans_op_mask, trans_label=trans_dec_op,
                conn_label=conn_ls, mode='valid', ret_loss=True
            )
        prod_n_logits, prod_e_logits, lg_act_logits, \
            conn_logits, conn_mask, trans_pred = preds

        syn_node_loss, syn_edge_loss, lg_act_loss, \
            conn_loss, trans_loss = losses

        loss = syn_node_loss + syn_edge_loss + lg_act_loss \
            + conn_loss + trans_loss

        loss_cur['syn_node_loss'].append(syn_edge_loss.item())
        loss_cur['syn_edge_loss'].append(syn_edge_loss.item())
        loss_cur['lg_act_loss'].append(lg_act_loss.item())
        loss_cur['conn_loss'].append(conn_loss.item())
        loss_cur['trans_loss'].append(trans_loss.item())
        loss_cur['all'].append(loss.item())


    return acc / total
