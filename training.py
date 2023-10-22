from data_utils import (
    convert_log_into_label, eval_by_node, eval_by_edge
)
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
import torch

from sklearn import metrics


def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):
    def f(x):
        if x >= warmup_iters:
            return 1
        alpha = float(x) / warmup_iters
        return warmup_factor * (1 - alpha) + alpha

    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


def train_sparse_edit(
    loader, model, optimizer, device, mode, verbose=True,
    warmup=True, reduction='mean', graph_level=True, pos_weight=1
):
    model = model.train()
    node_loss, edge_loss = [], []
    if warmup:
        warmup_iters = len(loader) - 1
        warmup_sher = warmup_lr_scheduler(optimizer, warmup_iters, 5e-2)
    for graph in tqdm(loader, ascii=True) if verbose else loader:
        graph = graph.to(device)

        node_pred, edge_pred, useful_mask, loss_node, loss_edge = \
            model(
                graph, mode, reduction, graph_level,
                ret_loss=True, pos_weight=pos_weight
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
    node_pd, node_lb, edge_pd, edge_lb = [], [], [], []
    for graph in tqdm(loader, ascii=True) if verbose else loader:
        graph = graph.to(device)
        with torch.no_grad():
            node_logs, edge_logs = model.predict_all_logits(graph)
            node_pred = convert_log_into_label(node_logs)
            edge_pred = convert_log_into_label(edge_logs)

        # comm_res = overall_acc(
        #     node_pred, edge_pred, graph.node_label, graph.edge_label
        # )

        # node_acc += comm_res[0]
        # edge_acc += comm_res[1]
        # node_cnt += comm_res[2]
        # edge_cnt += comm_res[3]

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

    node_pd = np.concatenate(node_pd, axis=0)
    node_lb = np.concatenate(node_lb, axis=0)
    edge_pd = np.concatenate(edge_pd, axis=0)
    edge_lb = np.concatenate(edge_lb, axis=0)

    result = {
        'common': {
            'node_roc': metrics.roc_auc_score(node_lb, node_pd),
            'edge_roc': metrics.roc_auc_score(edge_lb, edge_pd)
        },
        'by_node': {'cover': node_cov / tot, 'fit': node_fit / tot},
        'by_edge': {'cover': edge_cov / tot, 'fit': edge_fit / tot}
    }
    return result
