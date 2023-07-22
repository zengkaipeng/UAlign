import numpy as np
import torch
import torch.distributed as torch_dist

from data_utils import generate_tgt_mask
from enum import Enum
from tqdm import tqdm


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
        self.sum += val * num
        self.cnt += num
        self.avg = self.sum / self.cnt

    def all_reduce(self):
        infos = torch.FloatTensor([self.sum, self.cnt]).cuda()
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

    def all_reduct(self):
        for idx in range(len(self.metrics)):
            self.metrics[idx].all_reduce()

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


if __name__ == '__main__':
    X = MetricCollector(name='test')
    X.update(2)
    print(X)
