from model import evaluate_sparse
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
    loader, model, optimizer, device, empty_type=0, verbose=True, warmup=True,
    mode='together', reduction='mean', graph_level=True,
):
    model = model.train()
    node_loss, edge_loss = [], []
    if warmup:
        warmup_iters = len(loader) - 1
        warmup_sher = warmup_lr_scheduler(optimizer, warmup_iters, 5e-2)
    for graph in tqdm(loader) if verbose else loader:
        graph = graph.to(device)

        node_pred, edge_pred, loss_node, loss_edge = model(
            graph, mode, reduction, graph_level, ret_loss=True
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
    node_cover, node_fit, edge_fit, all_cov, all_fit, tot = [0] * 6
    for graph in (tqdm(loader) if verbose else loader):
        graph = graph.to(device)
        with torch.no_grad():
            node_pred, edge_pred = model(
                graph, mask_node='inference', ret_loss=False
            )

        metrics = evaluate_sparse(
            node_res=node_res, edge_res=edge_res, e_ptr=e_ptr,
            e_labels=torch.LongTensor(e_answer),
            node_ptr=graphs.ptr.tolist(), act_nodes=act_nodes
        )

        node_cover += metrics[0]
        node_fit += metrics[1]
        edge_fit += metrics[2]
        all_fit += metrics[3]
        all_cov += metrics[4]
        tot += metrics[5]
    return node_cover / tot, node_fit / tot, edge_fit / tot, \
        all_cov / tot, all_fit / tot
