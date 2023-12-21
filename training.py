from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
import torch
from utils.chemistry_parse import convert_res_into_smiles
from data_utils import (
    eval_by_batch, correct_trans_output,
    convert_log_into_label, eval_trans,
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
    losses, e_losses, AE_losses = [], [], []
    AH_losses, AC_losses = [], []
    if warmup:
        warmup_iters = len(loader) - 1
        warmup_sher = warmup_lr_scheduler(optimizer, warmup_iters, 5e-2)
    for graph in tqdm(loader):
        graph = graph.to(device)

        edge_loss, AE_loss, AH_loss, AC_loss = model(graph)

        loss = edge_loss + AE_loss + AH_loss + AC_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        e_losses.append(edge_loss.item())
        AE_losses.append(AE_loss.item())
        AC_losses.append(AC_loss.item())
        AH_losses.append(AH_loss.item())

        if warmup:
            warmup_sher.step()
    return {
        'total': np.mean(losses), 'edge_loss': np.mean(e_losses),
        'AE_loss': np.mean(AE_losses), 'AC_loss': np.mean(AC_losses),
        'AH_loss': np.mean(AH_losses)
    }


def eval_sparse_edit(loader, model, device):
    model = model.eval()
    e_accs, AE_accs, AH_accs, AC_accs = [[] for i in range(4)]
    for graph in tqdm(loader):
        graph = graph.to(device)
        with torch.no_grad():
            edge_logs, AE_logs, AH_logs, AC_logs = \
                model.eval_forward(graph, return_all=True)
            edge_pred = convert_edge_log_into_labels(
                edge_logs, graph.edge_index,
                mod='softmax', return_dict=False
            )

            AE_pred = convert_log_into_label(AE_logs, mod='sigmoid')
            AH_pred = convert_log_into_label(AH_logs, mod='sigmoid')
            AC_pred = convert_log_into_label(AC_logs, mod='sigmoid')

        edge_acc = eval_by_batch(
            edge_pred, graph.new_edge_types, graph.e_batch, True
        )
        AE_acc = eval_by_batch(AE_pred, graph.EdgeChange, graph.batch, True)
        AH_acc = eval_by_batch(AH_pred, graph.HChange, graph.batch, True)
        AC_acc = eval_by_batch(AC_pred, graph.ChargeChange, graph.batch, True)

        AE_accs.append(AE_acc)
        AH_accs.append(AH_acc)
        AC_accs.append(AC_acc)
        e_accs.append(edge_acc)

    e_accs = torch.cat(e_accs, dim=0).float()
    AE_accs = torch.cat(AE_accs, dim=0).float()
    AH_accs = torch.cat(AH_accs, dim=0).float()
    AC_accs = torch.cat(AC_accs, dim=0).float()
    return {
        'edge': e_accs.mean().item(), 'ChargeChange': AE_accs.mean().item(),
        'EdgeChange': AE_accs.mean().item(), 'HChange': AH_accs.mean().item()
    }


def train_overall(
    model, loader, optimizer, device, tokenizer,
    pad_token, warmup=True,
):
    Ael, Ahl, Acl = [], [], []
    enc_el, lg_act, conn, tras, al = [[] for _ in range(5)]
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

        AE_loss, AH_loss, AC_loss, syn_edge_loss, lg_act_loss, \
            conn_loss, trans_loss = losses

        loss = (AE_loss + AH_loss + AC_loss) + syn_edge_loss + \
            lg_act_loss + conn_loss + trans_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        Ael.append(AE_loss.item())
        Ahl.append(AH_loss.item())
        Acl.append(AC_loss.item())

        enc_el.append(syn_edge_loss.item())
        lg_act.append(lg_act_loss.item())
        conn.append(conn_loss.item())
        tras.append(trans_loss.item())
        al.append(loss.item())

        if warmup:
            warmup_sher.step()

    return {
        'AH_loss': np.mean(Ahl), 'AC_loss': np.mean(Acl),
        'AE_loss': np.mean(Ael), 'enc_edge_loss': np.mean(enc_el),
        'lg_act_loss': np.mean(lg_act), 'conn_loss': np.mean(conn),
        'trans_loss': np.mean(tras), 'all': np.mean(al)
    }


def eval_overall(
    model, loader, device, tokenizer, pad_token,
    end_token,
):
    model = model.eval()
    loss_cur = {
        'AE_loss': [], 'AH_loss': [], 'AC_loss': [],
        'lg_act_loss': [], 'enc_edge_loss': [],
        'trans_loss': [], 'conn_loss': [], 'all': []
    }
    metrics = {
        'synthon': {
            'edge_acc': [], 'HChange': [], 'ChargeChange': [],
            "EdgeChange": [],
        },
        'conn': {'lg_acc': [], 'conn_acc': []},
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

        # loss process

        AE_loss, AH_loss, AC_loss, syn_edge_loss, lg_act_loss, \
            conn_loss, trans_loss = losses

        loss = (AE_loss + AH_loss + AC_loss) + syn_edge_loss + \
            lg_act_loss + conn_loss + trans_loss

        loss_cur['AE_loss'].append(AE_loss.item())
        loss_cur['AH_loss'].append(AH_loss.item())
        loss_cur['AC_loss'].append(AC_loss.item())
        loss_cur['enc_edge_loss'].append(syn_edge_loss.item())
        loss_cur['lg_act_loss'].append(lg_act_loss.item())
        loss_cur['conn_loss'].append(conn_loss.item())
        loss_cur['trans_loss'].append(trans_loss.item())
        loss_cur['all'].append(loss.item())

        # pred process

        AE_logits, AH_logits, AC_logits, prod_e_logits, lg_act_logits, \
            conn_logits, conn_mask, trans_logits = preds

        AE_pred = convert_log_into_label(AE_logits, mod='sigmoid')
        AH_pred = convert_log_into_label(AH_logits, mod='sigmoid')
        AC_pred = convert_log_into_label(AC_logits, mod='sigmoid')

        edge_pred = convert_edge_log_into_labels(
            prod_e_logits, prod_graph.edge_index,
            mod='softmax', return_dict=False
        )

        lg_act_pred = convert_log_into_label(lg_act_logits, mod='sigmoid')
        conn_pred = convert_log_into_label(conn_logits, mod='softmax')
        trans_pred = convert_log_into_label(trans_logits, mod='softmax')
        trans_pred = correct_trans_output(trans_pred, end_idx, pad_idx)

        edge_acc = eval_by_batch(
            edge_pred, prod_graph.edge_label,
            prod_graph.e_batch, return_tensor=True
        )
        AE_acc = eval_by_batch(
            AE_pred, prod_graph.EdgeChange,
            prod_graph.batch, return_tensor=True
        )
        AH_acc = eval_by_batch(
            AH_pred, prod_graph.HChange,
            prod_graph.batch, return_tensor=True
        )
        AC_acc = eval_by_batch(
            AC_pred, prod_graph.ChargeChange,
            prod_graph.batch,  return_tensor=True
        )

        lg_act_acc = eval_by_batch(
            lg_act_pred, lg_graph.node_label, lg_graph.batch,
            return_tensor=True
        )

        conn_acc = eval_by_batch(
            conn_pred, conn_ls[conn_mask], conn_b[conn_mask],
            return_tensor=True
        )

        trans_acc = eval_trans(trans_pred, trans_dec_op, return_tensor=True)
        metrics['synthon']['edge_acc'].append(edge_acc)
        metrics['synthon']['HChange'].append(AH_acc)
        metrics['synthon']['EdgeChange'].append(AE_acc)
        metrics['synthon']['ChargeChange'].append(AC_acc)

        metrics['lg'].append(trans_acc)
        metrics['conn']['lg_acc'].append(lg_act_acc)
        metrics['conn']['conn_acc'].append(conn_acc)
        metrics['all'].append(lg_act_acc & conn_acc & edge_acc & trans_acc)

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
