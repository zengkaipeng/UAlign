from model import evaluate_sparse
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
import torch
from model import convert_graphs_into_decoder
from utils.chemistry_parse import convert_res_into_smiles
from data_utils import (
    eval_by_edge, eval_by_node, eval_by_graph,
    convert_log_into_label, convert_edge_log_into_labels
)
from sklearn import metrics
from data_utils import predict_synthon


def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):
    def f(x):
        if x >= warmup_iters:
            return 1
        alpha = float(x) / warmup_iters
        return warmup_factor * (1 - alpha) + alpha

    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


def train_sparse_edit(
    loader, model, optimizer, device, verbose=True,
    warmup=True, pos_weight=1
):
    model = model.train()
    node_loss, edge_loss = [], []
    if warmup:
        warmup_iters = len(loader) - 1
        warmup_sher = warmup_lr_scheduler(optimizer, warmup_iters, 5e-2)
    for graph in tqdm(loader, ascii=True) if verbose else loader:
        graph = graph.to(device)
        _, _, loss_node, loss_edge = model(
            graph, ret_loss=True, pos_weight=pos_weight
        )

        optimizer.zero_grad()
        (loss_node + loss_edge).backward()
        optimizer.step()

        node_loss.append(loss_node.item())
        edge_loss.append(loss_edge.item())

        if warmup:
            warmup_sher.step()
    return np.mean(node_loss), np.mean(edge_loss)


def eval_sparse_edit(loader, model, device, verbose=True):
    model = model.eval()
    node_cov, node_fit, edge_fit, edge_cov, tot = [0] * 5
    # node_acc, edge_acc, node_cnt, edge_cnt = [0] * 4
    g_nfit, g_ncov, g_efit, g_ecov = [0] * 4
    node_pd, node_lb, edge_pd, edge_lb = [], [], [], []
    for graph in tqdm(loader, ascii=True) if verbose else loader:
        graph = graph.to(device)
        with torch.no_grad():
            node_logs, edge_logs = model(graph, ret_loss=False)
            node_pred = convert_log_into_label(node_logs)
            edge_pred = convert_edge_log_into_labels(
                edge_logs, graph.edge_index, mod='sigmoid', return_dict=False
            )

        node_pd.append(node_logs.sigmoid().cpu().numpy())
        node_lb.append(graph.node_label.cpu().numpy())
        edge_pd.append(edge_logs.sigmoid().cpu().numpy())
        edge_lb.append(graph.edge_label.cpu().numpy())

        batch_size = graph.batch.max().item() + 1
        tot += batch_size

        cover, fit = eval_by_node(
            node_pred, edge_pred, graph.node_label, graph.edge_label,
            graph.batch, graph.e_batch, graph.edge_index
        )
        node_cov += cover
        node_fit += fit

        cover, fit = eval_by_edge(
            node_pred, edge_pred, graph.node_label, graph.edge_label,
            graph.batch, graph.e_batch, graph.edge_index, graph.ptr
        )
        edge_cov += cover
        edge_fit += fit

        g_metric = eval_by_graph(
            node_pred, edge_pred, graph.node_label,
            graph.edge_label, graph.batch, graph.e_batch,
        )

        g_nfit += g_metric[0]
        g_ncov += g_metric[1]
        g_efit += g_metric[2]
        g_ecov += g_metric[3]

    node_pd = np.concatenate(node_pd, axis=0)
    node_lb = np.concatenate(node_lb, axis=0)
    edge_pd = np.concatenate(edge_pd, axis=0)
    edge_lb = np.concatenate(edge_lb, axis=0)

    result = {
        'common': {
            'node_roc': metrics.roc_auc_score(node_lb, node_pd),
            'edge_roc': metrics.roc_auc_score(edge_lb, edge_pd)
        },
        'by_graph': {
            'node_cover': g_ncov / tot, 'node_fit': g_nfit / tot,
            'edge_cover': g_ecov / tot, 'edge_fit': g_efit / tot
        },
        'by_node': {'cover': node_cov / tot, 'fit': node_fit / tot},
        'by_edge': {'cover': edge_cov / tot, 'fit': edge_fit / tot}
    }
    return result


def train_overall(
    model, loader, optimizer, device, alpha=1, warmup=True,
    pos_weight=1, matching=True, aug_mode='none'
):
    enc_nl, enc_el, org_nl, org_el, pad_nl, pad_el = [[] for _ in range(6)]
    model = model.train()
    if warmup:
        warmup_iters = len(loader) - 1
        warmup_sher = warmup_lr_scheduler(optimizer, warmup_iters, 5e-2)

    for data in tqdm(loader, ascii=True):
        encoder_graph, decoder_graph, real_edge_type = data
        encoder_graph = encoder_graph.to(device)
        decoder_graph = decoder_graph.to(device)

        losses = model(
            encoder_graph, decoder_graph, real_edge_type,
            pos_weight=pos_weight, matching=matching, aug_mode=aug_mode
        )

        enc_n_loss, enc_e_loss, org_n_loss, org_e_loss, \
            pad_n_loss, pad_e_loss = losses

        loss = enc_n_loss + enc_e_loss + \
            alpha * (org_n_loss + org_e_loss) +\
            pad_n_loss + pad_e_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        enc_nl.append(enc_n_loss.item())
        enc_el.append(enc_e_loss.item())
        org_nl.append(org_n_loss.item())
        org_el.append(org_e_loss.item())
        pad_nl.append(pad_n_loss.item())
        pad_el.append(pad_e_loss.item())

        if warmup:
            warmup_sher.step()

    return {
        'enc_node_loss': np.mean(enc_nl), 'enc_edge_loss': np.mean(enc_el),
        'dec_org_n_loss': np.mean(org_nl), 'dec_org_e_loss': np.mean(org_el),
        'dec_pad_n_loss': np.mean(pad_nl), 'dec_pad_e_loss': np.mean(pad_el)
    }


def eval_overall(model, loader, device, mode='edge'):
    model, acc, total = model.eval(), 0, 0
    for data in tqdm(loader):
        encoder_graph, node_types, edge_types, smi = data
        encoder_graph = encoder_graph.to(device)
        batch_size = len(smi)
        total += batch_size

        with torch.no_grad():
            answer = model.predict(encoder_graoph, syn_mode=mode)
            enc_n_pred, enc_e_pred, pad_n_pred, pad_e_pred = answer

        synthon_nodes, synthon_edges = predict_synthon(
            batch_size=batch_size, n_pred=enc_n_pred, e_pred=enc_e_pred,
            graph=encoder_graph, n_types=node_types, e_types=edge_types
        )

        for i in range(batch_size):
            result = convert_res_into_smiles(
                synthon_nodes[i], synthon_edges[i],
                pad_n_pred[i], pad_e_pred[i]
            )
            acc += (smi[i] == result)

    return acc / total
