from data_utils import (
    convert_log_into_label, eval_by_node,
    eval_by_edge, overall_acc
)
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
import torch


def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):
    def f(x):
        if x >= warmup_iters:
            return 1
        alpha = float(x) / warmup_iters
        return warmup_factor * (1 - alpha) + alpha

    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


def train_sparse_edit(
    loader, model, optimizer, device, mode, verbose=True,
    warmup=True, reduction='mean', graph_level=True
):
    model = model.train()
    node_loss, edge_loss = [], []
    if warmup:
        warmup_iters = len(loader) - 1
        warmup_sher = warmup_lr_scheduler(optimizer, warmup_iters, 5e-2)
    for graph in tqdm(loader, ascii=True) if verbose else loader:
        graph = graph.to(device)

        node_pred, edge_pred, useful_mask, loss_node, loss_edge = \
            model(graph, mode, reduction, graph_level, ret_loss=True)

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
    node_acc, edge_acc, node_cnt, edge_cnt = [0] * 4
    for graph in tqdm(loader, ascii=True) if verbose else loader:
        graph = graph.to(device)
        with torch.no_grad():
            node_logs, edge_logs = model.predict_all_logits(graph)
            node_pred = node_logs.clone()
            edge_pred = edge_logs.clone()
            node_pred[node_pred >= 0] = 1

        batch_size = graph.batch.max().item() + 1

        metrics = evaluate_sparse(
            node_pred, edge_pred, graph.batch, graph.e_batch[useful_mask],
            graph.node_label, graph.edge_label[useful_mask], batch_size
        )

        node_cov += metrics[0]
        node_fit += metrics[1]
        edge_cov += metrics[2]
        edge_fit += metrics[3]
        all_cov += metrics[4]
        all_fit += metrics[5]
        tot += batch_size
    return node_cov / tot, node_fit / tot, edge_cov / tot, \
        edge_fit / tot, all_cov / tot, all_fit / tot
