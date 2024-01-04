from tqdm import tqdm
import numpy as np
import torch
from torch.nn.functional import cross_entropy
from data_utils import (
    generate_tgt_mask, correct_trans_output, eval_by_batch,
    convert_log_into_label, convert_edge_log_into_labels
)

from data_utils import eval_trans as data_eval_trans
from training import calc_trans_loss, loss_batch
import torch.distributed as torch_dist
from enum import Enum


class Summary(Enum):
    NONE, SUM, AVERAGE, COUNT = 0, 1, 2, 3


class MetricCollector(object):
    def __init__(self, name, type_fmt=':f', summary_type=Summary.AVERAGE):
        super(MetricCollector, self).__init__()
        self.name, self.type_fmt = name, type_fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val, self.sum, self.cnt, self.avg = [0] * 4

    def update(self, val, num=1):
        self.val = val
        self.sum += val
        self.cnt += num
        self.avg = self.sum / self.cnt

    def all_reduce(self, device):
        infos = torch.FloatTensor([self.sum, self.cnt]).to(device)
        torch_dist.all_reduce(infos, torch_dist.ReduceOp.SUM)
        self.sum, self.cnt = infos.tolist()
        self.avg = self.sum / self.cnt

    def __str__(self):
        return ''.join([
            '{name}: {val', self.type_fmt, '} avg: {avg', self.type_fmt, '}'
        ]).format(**self.__dict__)

    def summary(self):
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {cnt:.3f}'
        else:
            raise ValueError(f'Invaild summary type {self.summary_type} found')

        return fmtstr.format(**self.__dict__)

    def get_value(self):
        if self.summary_type is Summary.AVERAGE:
            return self.avg
        elif self.summary_type is Summary.SUM:
            return self.sum
        elif self.summary_type is Summary.COUNT:
            return self.cnt
        else:
            raise ValueError(
                f'Invaild summary type {self.summary_type} '
                'for get_value()'
            )


class MetricManager(object):
    def __init__(self, metrics):
        super(MetricManager, self).__init__()
        self.metrics = metrics

    def all_reduct(self, device):
        for idx in range(len(self.metrics)):
            self.metrics[idx].all_reduce(device)

    def summary_all(self, split_string='  '):
        return split_string.join(x.summary() for x in self.metrics)

    def get_all_value_dict(self):
        return {x.name: x.get_value() for x in self.metrics}


def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):
    def f(x):
        if x >= warmup_iters:
            return 1
        alpha = float(x) / warmup_iters
        return warmup_factor * (1 - alpha) + alpha

    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


def ddp_train_trans(
    loader, model, optimizer, tokenizer, device, node_fn,
    edge_fn, trans_fn, acc_fn, verbose=True, warmup=False,
    pad='<PAD>', unk='<UNK>', accu=1
):
    model = model.train()
    node_loss = MetricCollector('nloss', type_fmt=':.3f')
    edge_loss = MetricCollector('eloss', type_fmt=':.3f')
    tran_loss = MetricCollector('tloss', type_fmt=':.3f')
    xacc = MetricCollector('acc', type_fmt=':.3f')
    manager = MetricManager([node_loss, edge_loss, tran_loss, xacc])

    its, total_len = 1, len(loader)
    if warmup:
        warmup_iters = total_len - 1
        warmup_sher = warmup_lr_scheduler(optimizer, warmup_iters, 5e-2)

    iterx = tqdm(loader, ascii=True, desc='train') if verbose else loader
    for data in iterx:
        graphs, tgt = data
        graphs = graphs.to(device, non_blocking=True)
        tgt_idx = torch.LongTensor(tokenizer.encode2d(tgt))
        tgt_idx = tgt_idx.to(device, non_blocking=True)

        UNK_IDX = tokenizer.token2idx[unk]
        assert torch.all(tgt_idx != UNK_IDX).item(), \
            'Unseen tokens found, update tokenizer'

        tgt_input = tgt_idx[:, :-1]
        tgt_output = tgt_idx[:, 1:]

        pad_mask, sub_mask = generate_tgt_mask(
            tgt_input, tokenizer, pad, 'cpu'
        )
        pad_mask = pad_mask.to(device, non_blocking=True)
        sub_mask = sub_mask.to(device, non_blocking=True)

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

        node_loss.update(loss_node.item())
        edge_loss.update(loss_edge.item())
        tran_loss.update(loss_tran.item())

        with torch.no_grad():
            A, B = acc_fn(result, tgt_output)
            xacc.update(val=A, num=B)

        if warmup:
            warmup_sher.step()

        if verbose:
            iterx.set_postfix_str(manager.summary_all(split_string=','))

    return manager


def ddp_pretrain(
    loader, model, optimizer, device, tokenizer,
    pad_token, warmup, accu=1, verbose=False
):
    model = model.train()
    losses = MetricCollector('loss', type_fmt=':.3f')
    manager = MetricManager([losses])
    ignore_idx = tokenizer.token2idx[pad_token]
    its, total_len = 1, len(loader)
    if warmup:
        warmup_iters = len(loader) - 1
        warmup_sher = warmup_lr_scheduler(optimizer, warmup_iters, 5e-2)

    iterx = tqdm(loader, desc='train') if verbose else loader
    for graph, tran in iterx:
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

        if not warmup and accu > 1:
            loss = loss / accu
        loss.backward()

        if its % accu == 0 or its == total_len or warmup:
            optimizer.step()
            optimizer.zero_grad()
        its += 1

        losses.update(loss.item())

        if warmup:
            warmup_sher.step()

        if verbose:
            iterx.set_postfix_str(manager.summary_all())

    return manager


def ddp_eval_trans(
    loader, model, tran_fn, tokenizer, device,
    acc_fn, pad='<PAD>', verbose=True
):
    model = model.eval()
    tran_loss = MetricCollector('tloss', type_fmt=':.3f')
    xacc = MetricCollector('acc', type_fmt=':.3f')
    manager = MetricManager([tran_loss, xacc])

    iterx = tqdm(loader, ascii=True, desc='eval') if verbose else loader

    for data in iterx:
        graphs, tgt = data
        graphs = graphs.to(device, non_blocking=True)
        tgt_idx = torch.LongTensor(tokenizer.encode2d(tgt))
        tgt_idx = tgt_idx.to(device, non_blocking=True)
        tgt_input = tgt_idx[:, :-1]
        tgt_output = tgt_idx[:, 1:]

        pad_mask, sub_mask = generate_tgt_mask(
            tgt_input, tokenizer, pad, 'cpu'
        )

        pad_mask = pad_mask.to(device, non_blocking=True)
        sub_mask = sub_mask.to(device, non_blocking=True)

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
        xacc.update(val=A, num=B)
        tran_loss.update(loss_tran.item())
        if verbose:
            iterx.set_postfix_str(manager.summary_all())
    return manager


def ddp_preeval(
    model, loader, device, tokenizer, pad_token, end_token,
    verbose=False
):
    model = model.eval()

    trans_accs = MetricCollector('trans_acc', type_fmt=':.3f')
    manager = MetricManager([trans_accs])

    end_idx = tokenizer.token2idx[end_token]
    pad_idx = tokenizer.token2idx[pad_token]

    iterx = tqdm(loader, desc='eval') if verbose else loader

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
        A, B = data_eval_trans(trans_pred, trans_dec_op, False)
        trans_accs.update(val=A, num=B)
        if verbose:
            iterx.set_postfix_str(manager.summary_all())

    return manager


if __name__ == '__main__':
    X = MetricCollector(name='test')
    X.update(2)
    print(X)
