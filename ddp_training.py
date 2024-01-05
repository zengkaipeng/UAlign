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
    loader, model, optimizer, device, tokenizer, verbose=True, accu=1,
    warmup=False, pad='<PAD>', unk='<UNK>', label_smoothing=0.0
):
    model = model.train()
    acl = MetricCollector(name='AC', type_fmt=':.2f')
    ahl = MetricCollector(name='AH', type_fmt=':.2f')
    ael = MetricCollector(name='AE', type_fmt=':.2f')
    edge_loss = MetricCollector(name='edge', type_fmt=':.2f')
    tran_loss = MetricCollector(name='trans', type_fmt=':.2f')
    manager = MetricManager([acl, ahl, ael, edge_loss, tran_loss])

    its, total_len = 1, len(loader)
    if warmup:
        warmup_iters = len(loader) - 1
        warmup_sher = warmup_lr_scheduler(optimizer, warmup_iters, 5e-2)

    ignore_idx = tokenizer.token2idx[pad]
    iterx = tqdm(loader, desc='train') if verbose else loader
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

        edge_logs, AH_logs, AE_logs, AC_logs, result = model(
            graphs=graphs, tgt=tgt_input, tgt_mask=sub_mask,
            tgt_pad_mask=pad_mask,
        )

        AH_loss = loss_batch(AH_logs, graphs.HChange, graphs.batch)
        AC_loss = loss_batch(AC_logs, graphs.ChargeChange, graphs.batch)
        AE_loss = loss_batch(AE_logs, graphs.EdgeChange, graphs.batch)
        ed_loss = loss_batch(edge_logs, graphs.new_edge_types, graphs.e_batch)
        loss_tran = calc_trans_loss(
            result, tgt_output, ignore_idx, label_smoothing
        )

        loss = AC_loss + AH_loss + AE_loss + ed_loss + loss_tran
        if not warmup and accu > 1:
            loss = loss / accu
        loss.backward()

        if its % accu == 0 or its == total_len or warmup:
            optimizer.step()
            optimizer.zero_grad()
        its += 1

        acl.update(AC_loss.item())
        ahl.update(AH_loss.item())
        ael.update(AE_loss.item())
        edge_loss.update(ed_loss.item())
        tran_loss.update(loss_tran.item())

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
        graph = graph.to(device, non_blocking=True)
        tops = torch.LongTensor(tokenizer.encode2d(tran))
        tops = tops.to(device, non_blocking=True)
        trans_dec_ip = tops[:, :-1]
        trans_dec_op = tops[:, 1:]

        trans_op_mask, diag_mask = generate_tgt_mask(
            trans_dec_ip, tokenizer, pad_token, 'cpu'
        )

        trans_op_mask = trans_op_mask.to(device, non_blocking=True)
        diag_mask = diag_mask.to(device, non_blocking=True)

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
    loader, model, device, tokenizer,
    pad='<PAD>', end='<END>', verbose=True
):
    model = model.eval()
    pad_idx = tokenizer.token2idx[pad]
    end_idx = tokenizer.token2idx[end]
    tran_acc = MetricCollector(name='tran_acc', type_fmt=':.2f')
    eg_acc = MetricCollector(name='eg_acc', type_fmt=':.2f')
    ah_acc = MetricCollector(name='ah_acc', type_fmt=':.2f')
    ae_acc = MetricCollector(name='ae_acc', type_fmt=':.2f')
    ac_acc = MetricCollector(name='ac_acc', type_fmt=':.2f')

    manager = MetricManager([ac_acc, ah_acc, ae_acc, eg_acc, tran_acc])

    iterx = tqdm(loader, desc='eval') if verbose else loader

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
            edge_logs, AH_logs, AE_logs, AC_logs, result = model(
                graphs=graphs, tgt=tgt_input, tgt_mask=sub_mask,
                tgt_pad_mask=pad_mask,
            )

        AE_pred = convert_log_into_label(AE_logs, mod='softmax')
        AH_pred = convert_log_into_label(AH_logs, mod='softmax')
        AC_pred = convert_log_into_label(AC_logs, mod='softmax')

        edge_pred = convert_edge_log_into_labels(
            edge_logs, graphs.edge_index,
            mod='softmax', return_dict=False
        )

        trans_pred = convert_log_into_label(result, mod='softmax')
        trans_pred = correct_trans_output(trans_pred, end_idx, pad_idx)

        A, B = eval_by_batch(
            edge_pred, graphs.new_edge_types,
            graphs.e_batch, return_tensor=False
        )
        eg_acc.update(val=A, num=B)

        A, B = eval_by_batch(
            AE_pred, graphs.EdgeChange,
            graphs.batch, return_tensor=False
        )
        ae_acc.update(val=A, num=B)

        A, B = eval_by_batch(
            AH_pred, graphs.HChange,
            graphs.batch, return_tensor=False
        )
        ah_acc.update(val=A, num=B)

        A, B = eval_by_batch(
            AC_pred, graphs.ChargeChange,
            graphs.batch,  return_tensor=False
        )
        ac_acc.update(val=A, num=B)

        A, B = data_eval_trans(trans_pred, tgt_output, return_tensor=False)
        tran_acc.update(val=A, num=B)

        if verbose:
            iterx.set_postfix_str(manager.summary_all(split_string=','))

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

    for graph, tran in iterx:
        graph = graph.to(device, non_blocking=True)
        tops = torch.LongTensor(tokenizer.encode2d(tran))
        tops = tops.to(device, non_blocking=True)
        trans_dec_ip = tops[:, :-1]
        trans_dec_op = tops[:, 1:]

        trans_op_mask, diag_mask = generate_tgt_mask(
            trans_dec_ip, tokenizer, pad_token, 'cpu'
        )

        trans_op_mask = trans_op_mask.to(device, non_blocking=True)
        diag_mask = diag_mask.to(device, non_blocking=True)

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
