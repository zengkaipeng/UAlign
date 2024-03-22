from tqdm import tqdm
import numpy as np
import torch
from torch.nn.functional import cross_entropy
from data_utils import (
    generate_tgt_mask, correct_trans_output, eval_by_batch,
    convert_log_into_label, convert_edge_log_into_labels
)

from data_utils import eval_trans as data_eval_trans


def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):
    def f(x):
        if x >= warmup_iters:
            return 1
        alpha = float(x) / warmup_iters
        return warmup_factor * (1 - alpha) + alpha

    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


def loss_batch(logs, label, batch):
    losses = cross_entropy(logs, label, reduction='none')
    all_x = torch.zeros(batch.max().item() + 1).to(losses)
    all_x.index_add_(dim=0, source=losses, index=batch)
    return all_x.mean()


def calc_trans_loss(trans_pred, trans_lb, ignore_index, lbsm=0.0):
    batch_size, maxl, num_c = trans_pred.shape
    trans_pred = trans_pred.reshape(-1, num_c)
    trans_lb = trans_lb.reshape(-1)

    losses = cross_entropy(
        trans_pred, trans_lb, reduction='none',
        ignore_index=ignore_index, label_smoothing=lbsm
    )
    losses = losses.reshape(batch_size, maxl)
    loss = torch.mean(torch.sum(losses, dim=-1))
    return loss


def pretrain(
    loader, model, optimizer, device, tokenizer,
    pad_token, warmup, accu=1, label_smoothing=0
):
    model, losses = model.train(), []
    ignore_idx = tokenizer.token2idx[pad_token]
    its, total_len = 1, len(loader)
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

        loss = calc_trans_loss(
            trans_logs, trans_dec_op, ignore_idx, label_smoothing
        )

        if not warmup and accu > 1:
            loss = loss / accu
        loss.backward()

        if its % accu == 0 or its == total_len or warmup:
            optimizer.step()
            optimizer.zero_grad()
        its += 1

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
