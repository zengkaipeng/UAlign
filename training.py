from model import evaluate_sparse
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
    loader, model, optimizer, device, empty_type=0,
    verbose=True, warmup=True, mode='together'
):
    model = model.train()
    node_loss, edge_loss = [], []
    for data in tqdm(loader) if verbose else loader:
        if warmup:
            warmup_iters = len(loader) - 1
            warmup_sher = warmup_lr_scheduler(optimizer, warmup_iters, 5e-2)
        if len(data) == 6:
            graphs, r_cls, node_label, e_type, act_nodes, e_map = data
        else:
            graphs, node_label, e_type, act_nodes, e_map = data
            r_cls = None

        graphs = graphs.to(device)
        node_label = node_label.to(device)
        if r_cls is not None:
            r_cls = r_cls.to(device)

        node_res, edge_res, e_answer, e_ptr, new_act_nodes = model(
            graphs=graphs, act_nodes=act_nodes,  mode=mode, e_types=e_type,
            edge_map=e_map, empty_type=empty_type
        )

        loss_node = F.cross_entropy(node_res, node_label)
        if edge_res is not None:
            edge_labels = torch.LongTensor(e_answer).to(device)
            loss_edge = F.cross_entropy(edge_res, edge_labels)
        else:
            loss_edge = torch.tensor(0.0).to(device)

        optimizer.zero_grad()
        (loss_node + loss_edge).back_ward()
        optimizer.step()

        node_loss.append(loss_node.item())
        edge_loss.append(loss_edge.item())

        if warmup:
            warmup_sher.step()
    return np.mean(node_loss), np.mean(edge_loss)


def eval_sparse_edit(loader, model, device, verbose=True):
    model = model.eval()
    node_cover, node_fit, edge_fit, all_cov, all_fit, tot = [0] * 6
    for data in (tqdm(loader) if verbose else loader):
        if len(data) == 6:
            graphs, r_cls, node_label, e_type, act_nodes, e_map = data
        else:
            graphs, node_label, e_type, act_nodes, e_map = data
            r_cls = None

        graphs = graphs.to(device)
        node_label = node_label.to(device)
        if r_cls is not None:
            r_cls = r_cls.to(device)
        with torch.no_grad():
            node_res, edge_res, e_answer, e_ptr, new_act_nodes = model(
                graphs=graphs, act_nodes=None, mode='inference',
                e_types=e_type, edge_map=e_map, empty_type=empty_type
            )
        metrics = evaluate_sparse(
            node_res=node_res, pred_edge=edge_res, e_ptr=e_ptr,
            e_labels=torch.LongTensor(e_answer),
            node_ptr=graphs.ptr.tolist(), act_nodes=act_nodes
        )

        node_cover += metrics[0]
        node_fit += metrics[1]
        edge_fit += metric[2]
        all_cov += metric[3]
        all_fit += metric[4]
        tot += metric[5]
    return node_cover / tot, node_fit / tot, edge_fit / tot, \
        all_cov / tot, all_fit / tot
