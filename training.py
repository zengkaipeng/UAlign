from tqdm import tqdm
import numpy as np
import torch
from torch.nn.functional import cross_entropy
from data_utils import (
    generate_tgt_mask, correct_trans_output, 
    convert_log_into_label
)

from data_utils import eval_trans as data_eval_trans

def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):
    def f(x):
        if x >= warmup_iters:
            return 1
        alpha = float(x) / warmup_iters
        return warmup_factor * (1 - alpha) + alpha

    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


def train_trans(
    loader, model, optimizer, device, tokenizer, node_fn,
    edge_fn, trans_fn, acc_fn, verbose=True, warmup=False,
    pad='<PAD>', unk='<UNK>', accu=1
):
    model, ele_acc, ele_total = model.train(), 0, 0
    node_loss, edge_loss, tran_loss = [], [], []
    its, total_len = 1, len(loader)
    if warmup:
        warmup_iters = len(loader) - 1
        warmup_sher = warmup_lr_scheduler(optimizer, warmup_iters, 5e-2)
    for data in tqdm(loader) if verbose else loader:
        graphs, tgt = data
        graphs = graphs.to(device)
        tgt_idx = torch.LongTensor(tokenizer.encode2d(tgt)).to(device)
        UNK_IDX = tokenizer.token2idx[unk]
        assert torch.all(tgt_idx != UNK_IDX).item(), \
            'Unseen tokens found, update tokenizer'

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

        loss = loss_node + loss_edge + loss_tran
        if not warmup and accu > 1:
            loss = loss / accu
        loss.backward()

        if its % accu == 0 or its == total_len or warmup:
            optimizer.step()
            optimizer.zero_grad()
        its += 1

        node_loss.append(loss_node.item())
        edge_loss.append(loss_edge.item())
        tran_loss.append(loss_tran.item())

        with torch.no_grad():
            A, B = acc_fn(result, tgt_output)
            ele_acc, ele_total = ele_acc + A, ele_total + B

        if warmup:
            warmup_sher.step()
    return np.mean(node_loss), np.mean(edge_loss), \
        np.mean(tran_loss), ele_acc / ele_total


def eval_trans(
    loader, model, device, tran_fn, tokenizer,
    acc_fn, pad='<PAD>', verbose=True
):
    model = model.eval()
    tran_loss, ele_total, ele_acc = [], 0, 0
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
            A, B = acc_fn(result, tgt_output)
            ele_acc, ele_total = ele_acc + A, ele_total + B
        tran_loss.append(loss_tran.item())
    return np.mean(tran_loss), ele_acc / ele_total


def calc_trans_loss(trans_pred, trans_lb, ignore_index):
    batch_size, maxl, num_c = trans_pred.shape
    trans_pred = trans_pred.reshape(-1, num_c)
    trans_lb = trans_lb.reshape(-1)

    losses = cross_entropy(
        trans_pred, trans_lb, reduction='none',
        ignore_index=ignore_index
    )
    losses = losses.reshape(batch_size, maxl)
    loss = torch.mean(torch.sum(losses, dim=-1))
    return loss


def pretrain(
    loader, model, optimizer, device, tokenizer,
    pad_token, warmup
):
    model, losses = model.train(), []
    ignore_idx = tokenizer.token2idx[pad_token]
    if warmup:
        warmup_iters = len(loader) - 1
        warmup_sher = warmup_lr_scheduler(optimizer, warmup_iters, 5e-2)
    for graph, tran in tqdm(loader):
        graph = graph.to(device)

        tops = torch.LongTensor(tokenizer.encode2d(tran)).to(device)
        trans_dec_ip = tops[:, :-1]
        trans_dec_op = tops[:, 1:]

        trans_op_mask, diag_mask = generate_tgt_mask(
            trans_dec_ip, tokenizer, pad_token, device=device
        )

        trans_logs = model(
            graphs=graph, tgt=trans_dec_ip, tgt_mask=diag_mask,
            tgt_pad_mask=trans_op_mask
        )

        loss = calc_trans_loss(trans_logs, trans_dec_op, ignore_idx)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if warmup:
            warmup_sher.step()

    return np.mean(losses)


def preeval(model, loader, device, tokenizer, pad_token, end_token):
    model, trans_accs = model.eval(), []
    end_idx = tokenizer.token2idx[end_token]
    pad_idx = tokenizer.token2idx[pad_token]

    for graph, tran in tqdm(loader):
        graph = graph.to(device)
        tops = torch.LongTensor(tokenizer.encode2d(tran)).to(device)
        trans_dec_ip = tops[:, :-1]
        trans_dec_op = tops[:, 1:]

        trans_op_mask, diag_mask = generate_tgt_mask(
            trans_dec_ip, tokenizer, pad_token, device=device
        )
        with torch.no_grad():
            trans_logs = model(
                graphs=graph, tgt=trans_dec_ip, tgt_mask=diag_mask,
                tgt_pad_mask=trans_op_mask
            )
            trans_pred = convert_log_into_label(trans_logs, mod='softmax')
            trans_pred = correct_trans_output(trans_pred, end_idx, pad_idx)
        trans_acc = data_eval_trans(trans_pred, trans_dec_op, True)
        trans_accs.append(trans_acc)

    trans_accs = torch.cat(trans_accs, dim=0).float()
    return trans_accs.mean().item()
