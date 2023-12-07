from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
import torch
from utils.chemistry_parse import convert_res_into_smiles
from data_utils import (
    eval_by_batch, correct_trans_output,
    convert_log_into_label, eval_trans,
    convert_edge_log_into_labels, eval_conn
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
    losses = []
    if warmup:
        warmup_iters = len(loader) - 1
        warmup_sher = warmup_lr_scheduler(optimizer, warmup_iters, 5e-2)
    for graph in tqdm(loader):
        graph = graph.to(device)
        _, loss = model(graph, ret_loss=True)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if warmup:
            warmup_sher.step()
    return np.mean(losses)


def eval_sparse_edit(loader, model, device):
    model = model.eval()
    accs = []
    for graph in tqdm(loader):
        graph = graph.to(device)
        with torch.no_grad():
            edge_logs = model(graph, ret_loss=False)
            edge_pred = convert_edge_log_into_labels(
                edge_logs, graph.edge_index,
                mod='softmax', return_dict=False
            )
        acc = eval_by_batch(
            edge_pred, graph.new_edge_types, 
            graph.e_batch, return_tensor=True
        )
        accs.append(acc)

    return torch.cat(accs, dim=0).float().mean().item()


def train_overall(
    model, loader, optimizer, device, tokenizer,
    pad_token, warmup=True,
):
    enc_nl, enc_el, lg_act, conn, tras, al = [[] for _ in range(6)]
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

        tips = torch.LongTensor(tokenizer.encode2d(tips)).to(device)
        tops = torch.LongTensor(tokenizer.encode2d(tops)).to(device)

        trans_ip_mask = tips == pad_idx

        trans_dec_ip = tops[:, :-1]
        trans_dec_op = tops[:, 1:]

        trans_op_mask, diag_mask = generate_tgt_mask(
            trans_dec_ip, tokenizer, pad_token, device=device
        )
        if grxn is not None:
            grxn = grxn.to(device)

        losses = model(
            prod_graph=prod_graph, lg_graph=lg_graph, trans_ip=tips,
            conn_edges=conn_es, conn_batch=conn_b, trans_op=trans_dec_ip,
            graph_rxn=grxn, pad_idx=pad_idx,  trans_op_mask=diag_mask,
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
        # for x, y in model.named_parameters():
        #     print(x, y.grad)
        # exit()
        optimizer.step()

        enc_nl.append(syn_node_loss.item())
        enc_el.append(syn_edge_loss.item())
        lg_act.append(lg_act_loss.item())
        conn.append(conn_loss.item())
        tras.append(trans_loss.item())
        al.append(loss.item())

        if warmup:
            warmup_sher.step()

    return {
        'enc_node_loss': np.mean(enc_nl), 'enc_edge_loss': np.mean(enc_el),
        'lg_act_loss': np.mean(lg_act), 'conn_loss': np.mean(conn),
        'trans_loss': np.mean(tras), 'all': np.mean(al)
    }


def eval_overall(
    model, loader, device, tokenizer, pad_token,
    end_token,
):
    model = model.eval()
    loss_cur = {
        'syn_node_loss': [], 'syn_edge_loss': [], 'all': [],
        'lg_act_loss': [], 'conn_loss': [], 'trans_loss': []
    }
    metrics = {
        'synthon': {'node_acc': [], 'break_acc': [], 'break_cover': []},
        'conn': {'lg_cov': [], 'lg_acc': [], 'conn_cov': [], 'conn_acc': []},
        'all': [], 'lg': [],
    }
    pad_idx = tokenizer.token2idx[pad_token]
    end_idx = tokenizer.token2idx[end_token]
    for data in tqdm(loader):
        prod_graph, lg_graph, conn_es, conn_ls, conn_b, tips, tops, grxn = data
        prod_graph = prod_graph.to(device)
        lg_graph = lg_graph.to(device)
        conn_es = conn_es.to(device)
        conn_ls = conn_ls.to(device)
        conn_b = conn_b.to(device)

        tips = torch.LongTensor(tokenizer.encode2d(tips)).to(device)
        tops = torch.LongTensor(tokenizer.encode2d(tops)).to(device)
        batch_size = prod_graph.batch.max().item() + 1

        trans_ip_mask = tips == pad_idx
        trans_dec_ip = tops[:, :-1]
        trans_dec_op = tops[:, 1:]

        trans_op_mask, diag_mask = generate_tgt_mask(
            trans_dec_ip, tokenizer, pad_token, device=device
        )
        if grxn is not None:
            grxn = grxn.to(device)

        with torch.no_grad():
            preds, losses = model(
                prod_graph=prod_graph, lg_graph=lg_graph, trans_ip=tips,
                conn_edges=conn_es, conn_batch=conn_b, trans_op=trans_dec_ip,
                graph_rxn=grxn, pad_idx=pad_idx,  trans_op_mask=diag_mask,
                trans_ip_key_padding=trans_ip_mask,
                trans_op_key_padding=trans_op_mask, trans_label=trans_dec_op,
                conn_label=conn_ls, mode='valid', return_loss=True
            )

        # losses process
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

        # pred process

        prod_n_logits, prod_e_logits, lg_act_logits, \
            conn_logits, conn_mask, trans_logits = preds

        node_pred = convert_log_into_label(prod_n_logits, mod='softmax')
        edge_pred = convert_edge_log_into_labels(
            prod_e_logits, prod_graph.edge_index,
            mod='softmax', return_dict=False
        )

        lg_act_pred = convert_log_into_label(lg_act_logits, mod='sigmoid')
        conn_pred = convert_log_into_label(conn_logits, mod='sigmoid')
        trans_pred = convert_log_into_label(trans_logits, mod='softmax')
        trans_pred = correct_trans_output(trans_pred, end_idx, pad_idx)

        node_acc, break_acc, break_cover = eval_by_graph(
            node_pred, edge_pred, prod_graph.node_label, prod_graph.edge_label,
            prod_graph.batch, prod_graph.e_batch, return_tensor=True
        )
        lg_acc, lg_cover, conn_acc, conn_cover = eval_conn(
            lg_pred=lg_act_pred, lg_label=lg_graph.node_label,
            lg_batch=lg_graph.batch, conn_pred=conn_pred,
            conn_lable=conn_ls[conn_mask], conn_batch=conn_b[conn_mask],
            return_tensor=True, batch_size=batch_size
        )
        trans_acc = eval_trans(trans_pred, trans_dec_op, return_tensor=True)
        metrics['synthon']['node_acc'].append(node_acc)
        metrics['synthon']['break_cover'].append(break_cover)
        metrics['synthon']['break_acc'].append(break_acc)
        metrics['lg'].append(trans_acc)
        metrics['conn']['lg_acc'].append(lg_acc)
        metrics['conn']['lg_cov'].append(lg_cover)
        metrics['conn']['conn_acc'].append(conn_acc)
        metrics['conn']['conn_cov'].append(conn_cover)
        metrics['all'].append(conn_cover & break_cover & trans_acc)

    loss_cur = {k: np.mean(v) for k, v in loss_cur.items()}

    for k, v in metrics.items():
        if isinstance(v, list):
            metrics[k] = torch.cat(v, dim=0).float().mean().item()
        else:
            metrics[k] = {
                x: torch.cat(y, dim=0).float().mean().item()
                for x, y in v.items()
            }
    return loss_cur, metrics
