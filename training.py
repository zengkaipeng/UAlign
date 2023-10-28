from model import evaluate_sparse
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
import torch
from model import convert_graphs_into_decoder
from utils.chemistry_parse import convert_res_into_smiles
from data_utils import (
    eval_by_edge, eval_by_node, eval_by_graph,
    convert_log_into_label
)


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
            edge_pred = convert_log_into_label(edge_logs)

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
    model, loader, optimizer, device, mode, alpha=1,
    warmup=True, reduction='mean', graph_level=True
):
    model = model.train()
    overall_loss = []
    if warmup:
        warmup_iters = len(loader) - 1
        warmup_sher = warmup_lr_scheduler(optimizer, warmup_iters, 5e-2)

    for data in tqdm(loader, ascii=True):
        encoder_graph, decoder_graph, real_edge_type = data
        encoder_graph = encoder_graph.to(device)
        decoder_graph = decoder_graph.to(device)

        loss = model(
            encoder_graph, decoder_graph, encoder_mode=mode,
            edge_types=real_edge_type, reduction=reduction,
            graph_level=graph_level, alpha=alpha
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        overall_loss.append(loss.item())
        if warmup:
            warmup_sher.step()

    return np.mean(overall_loss)


def eval_overall(model, loader, device, pad_num):
    model, acc, total = model.eval(), 0, 0
    for data in tqdm(loader):
        encoder_graph, node_types, edge_types, smi = data
        encoder_graph = encoder_graph.to(device)
        batch_size = encoder_graph.batch.max().item() + 1
        total += batch_size
        with torch.no_grad():
            synthons = model.encoder.predict_into_graphs(encoder_graph)
            memory, mem_pad_mask = model.encoder.make_memory(encoder_graph)

        synt_nodes, synt_edges = [], []
        for idx, synt in enumerate(synthons):
            this_node_types = node_types[idx]
            this_edge_types = edge_types[idx]
            # print(this_node_types, this_edge_types)
            synt_nodes.append({
                x: this_node_types[x]
                for x in range(synt['x'].shape[0])
            })

            remain_edges = set({
                (x.item(), y.item()) if x.item() < y.item() else
                (y.item(), x.item()) for x, y in synt['res_edge'].T
            })
            synt_edges.append({x: this_edge_types[x] for x in remain_edges})

        decoder_graph = convert_graphs_into_decoder(synthons, pad_num)
        # print(decoder_graph)

        with torch.no_grad():
            pad_nodes, pad_edges = model.decoder.predict_paddings(
                decoder_graph.to(device), memory, mem_pad_mask
            )

        for i in range(batch_size):
            # print(synt_nodes[i], synt_edges[i], pad_nodes[i], pad_edges[i])
            # exit()
            x = synt_edges[i].keys()
            y = pad_edges[i].keys()

            result = convert_res_into_smiles(
                synt_nodes[i], synt_edges[i], pad_nodes[i], pad_edges[i]
            )
            acc += (smi[i] == result)

    return acc / total
