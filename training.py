from model import evaluate_sparse
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
import torch
from model import convert_graphs_into_decoder
from utils.chemistry_parse import convert_res_into_smiles


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
    node_cov, node_fit, edge_fit, edge_cov, all_cov, all_fit, tot = [0] * 7
    for graph in tqdm(loader, ascii=True) if verbose else loader:
        graph = graph.to(device)
        with torch.no_grad():
            node_pred, edge_pred, useful_mask = model(
                graph, mask_mode='inference', ret_loss=False
            )
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

        with torch.no_grad():
            pad_nodes, pad_edges = model.decoder.predict_paddings(
                decoder_graph, memory, mem_pad_mask
            )

        for i in range(batch_size):
            print(synt_nodes[i], synt_edges[i], pad_nodes[i], pad_edges[i])
            exit()
            result = convert_res_into_smiles(
                synt_nodes[i], synt_edges[i], pad_nodes[i], pad_edges[i]
            )
            acc += (smi[i] == result)

    return acc / curr
