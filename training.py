from tqdm import tqdm
import numpy as np
import torch
from data_utils import generate_tgt_mask


def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):
    def f(x):
        if x >= warmup_iters:
            return 1
        alpha = float(x) / warmup_iters
        return warmup_factor * (1 - alpha) + alpha

    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


def train_trans(
    loader, model, optimizer, device, tokenizer, node_fn,
    edge_fn, trans_fn, verbose=True, warmup=False, pad='<PAD>'
):
    model = model.train()
    node_loss, edge_loss, tran_loss = [], [], []
    if warmup:
        warmup_iters = len(loader) - 1
        warmup_sher = warmup_lr_scheduler(optimizer, warmup_iters, 5e-2)
    for data in tqdm(loader) if verbose else loader:
        graphs, tgt = data
        graphs = graphs.to(device)
        tgt_idx = torch.LongTensor(tokenizer.encode2d(tgt)).to(device)
        tgt_input = tgt_idx[:, :-1]
        tgt_output = tgt_idx[:, 1:]

        pad_mask, sub_mask = generate_tgt_mask(
            tgt_input, tokenizer, pad, device
        )

        result, node_res, edge_res = model(
            graphs=graphs, tgt=tgt_input, tgt_mask=sub_mask,
            tgt_pad_mask=pad_mask, pred_core=True
        )

        loss_node = node_fn(node_res, graphs.node_label)
        loss_edge = edge_fn(edge_res, graphs.edge_label)
        loss_tran = trans_fn(
            result.reshape(-1, result.shape[-1]),
            tgt_output.reshape(-1)
        )

        optimizer.zero_grad()
        (loss_node + loss_edge + loss_tran).backward()
        optimizer.step()

        node_loss.append(loss_node.item())
        edge_loss.append(loss_edge.item())
        tran_loss.append(loss_tran.item())

        if warmup:
            warmup_sher.step()
    return np.mean(node_loss), np.mean(edge_loss), np.mean(tran_loss)


def eval_trans(
    loader, model, device, tran_fn, tokenizer,
    pad='<PAD>', verbose=True
):
    model = model.eval()
    tran_loss = []
    for data in tqdm(loader) if verbose else loader:
        graphs, tgt = data
        graphs = graphs.to(device)
        tgt_idx = torch.LongTensor(tokenizer.encode2d(tgt)).to(device)
        tgt_input = tgt_idx[:, :-1]
        tgt_output = tgt_idx[:, 1:]

        pad_mask, sub_mask = generate_tgt_mask(
            tgt_input, tokenizer, pad, device
        )

        with torch.no_grad():
            result = model(
                graphs=graphs, tgt=tgt_input, tgt_mask=sub_mask,
                tgt_pad_mask=pad_mask, pred_core=False
            )
            loss_tran = tran_fn(
                result.reshape(-1, result.shape[-1]),
                tgt_output.reshape(-1)
            )
        tran_loss.append(loss_tran.item())
    return np.mean(tran_loss)


def evaluate_result(gt, results, tokenizer, tops):
    res, tpos = tokenizer.decode2d(results), max(top) + 1
    ax, gt = np.array(tops), ''.join(gt[1: -1])
    for idx, t in enumerate(res):
        if t == gt:
            tpos = idx + 1
            break

    return (ax >= tpos)
