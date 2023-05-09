from model import get_labels, evaluate_sparse
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np


def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):
    def f(x):
        if x >= warmup_iters:
            return 1
        alpha = float(x) / warmup_iters
        return warmup_factor * (1 - alpha) + alpha

    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


def train_sparse_edit(
    loader, model, optimizer, verbose=True,
    warmup=True, mode='together'
):
    node_loss, edge_loss = [], []
    for data in tqdm(loader) if verbose else loader:
        if warmup:
            warmup_iters = len(loader) - 1
            warmup_sher = warmup_lr_scheduler(optimizer, warmup_iters, 5e-2)
        if len(data) == 6:
            graphs, node_label, num_l, num_e, e_type, act_nodes = data
            r_cls = None
        else:
            graphs, r_cls, node_label, num_l, num_e, e_type, act_nodes = data

        node_res, edge_res, new_act_nodes = model(
            graphs=graphs, act_nodes=act_nodes, num_nodes=num_l, mode=mode,
            num_edges=num_e,  return_feat=False, rxn_class=r_cls
        )

        loss_node = F.cross_entropy(node_res, node_label)
        edge_labels = get_labels(new_act_nodes, e_type)
        loss_edge = F.cross_entropy(edge_res, edge_labels)

        optimizer.zero_grad()
        (loss_node + loss_edge).back_ward()
        optimizer.step()

        node_loss.append(loss_node.item())
        edge_loss.append(loss_edge.item())

        if warmup:
            warmup_sher.step()
    return np.mean(node_loss), np.mean(edge_loss)


def eval_sparse_edit(loader, model, verbose=True):
    node_cover, node_fit, edge_fit, all_cov, all_fit, tot = [0] * 6
    for data in tqdm(loader) if verbose else loader:
        if len(data) == 6:
            graphs, node_label, num_l, num_e, e_type, act_nodes = data
            r_cls = None
        else:
            graphs, r_cls, node_label, num_l, num_e, e_type, act_nodes = data
        node_res, edge_res, used_nodes = model(
            graphs=graphs, num_nodes=num_l, num_edges=num_e,
            mode='inference', return_feat=False, rxn_class=r_cls
        )
        metrics = evaluate_sparse(
            node_res=node_res, pred_edge=edge_res, num_nodes=num_l,
            num_edges=num_e, edge_types=edge_type, act_nodes=act_nodes,
            used_nodes=used_nodes
        )

        node_cover += metrics[0]
        node_fit += metrics[1]
        edge_fit += metric[2]
        all_cov += metric[3]
        all_fit += metric[4]
        tot += metric[5]
    return node_cover / tot, node_fit / tot, edge_fit / tot, \
        all_cov / tot, all_fit / tot
