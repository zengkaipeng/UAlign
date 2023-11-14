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
    node_acc, break_acc, break_cover, tot = [0] * 4
    for graph in tqdm(loader, ascii=True) if verbose else loader:
        graph = graph.to(device)
        with torch.no_grad():
            node_logs, edge_logs = model(graph, ret_loss=False)
            node_pred = convert_log_into_label(node_logs, mod='softmax')
            edge_pred = convert_edge_log_into_labels(
                edge_logs, graph.edge_index, mod='sigmoid', return_dict=False
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
        # print(decoder_graph.node_class)

        losses = model(
            encoder_graph, decoder_graph, real_edge_type,
            pos_weight=pos_weight, use_matching=matching, aug_mode=aug_mode
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
            answer = model.predict(encoder_graph, syn_mode=mode)
            enc_n_pred, enc_e_pred, pad_n_pred, pad_e_pred = answer

        synthon_nodes, synthon_edges = predict_synthon(
            n_pred=enc_n_pred, e_pred=enc_e_pred, graph=encoder_graph,
            n_types=node_types, e_types=edge_types
        )

        for i in range(batch_size):
            result = convert_res_into_smiles(
                synthon_nodes[i], synthon_edges[i],
                {k: v for k, v in pad_n_pred[i].items() if v != 0},
                {k: v for k, v in pad_e_pred[i].items() if v != 0}
            )
            # print(result, smi[i])
            acc += (smi[i] == result)

    return acc / total
