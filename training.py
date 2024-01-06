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


def train_trans(
    loader, model, optimizer, device, tokenizer, verbose=True, accu=1,
    warmup=False, pad='<PAD>', unk='<UNK>', label_smoothing=0.0
):
    model = model.train()
    acl, ahl, ael, edge_loss, tran_loss = [[] for i in range(5)]
    its, total_len = 1, len(loader)
    if warmup:
        warmup_iters = len(loader) - 1
        warmup_sher = warmup_lr_scheduler(optimizer, warmup_iters, 5e-2)

    ignore_idx = tokenizer.token2idx[pad]
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

        acl.append(AC_loss.item())
        ahl.append(AH_loss.item())
        ael.append(AE_loss.item())
        edge_loss.append(ed_loss.item())
        tran_loss.append(loss_tran.item())

        if warmup:
            warmup_sher.step()
    return {
        'ChargeChange': np.mean(acl), 'HChange': np.mean(ahl),
        'EdgeChange': np.mean(ael), 'Edge': np.mean(edge_loss),
        'trans': np.mean(tran_loss)
    }


def eval_trans(
    loader, model, device, tokenizer,
    pad='<PAD>', end='<END>', verbose=True
):
    model = model.eval()
    pad_idx = tokenizer.token2idx[pad]
    end_idx = tokenizer.token2idx[end]
    tran_acc, eg_acc, ah_acc, ae_acc, ac_acc = [[] for i in range(5)]
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

        edge_acc = eval_by_batch(
            edge_pred, graphs.new_edge_types,
            graphs.e_batch, return_tensor=True
        )
        AE_acc = eval_by_batch(
            AE_pred, graphs.EdgeChange,
            graphs.batch, return_tensor=True
        )
        AH_acc = eval_by_batch(
            AH_pred, graphs.HChange,
            graphs.batch, return_tensor=True
        )
        AC_acc = eval_by_batch(
            AC_pred, graphs.ChargeChange,
            graphs.batch,  return_tensor=True
        )

        trans_acc = data_eval_trans(trans_pred, tgt_output, return_tensor=True)

        tran_acc.append(trans_acc)
        ae_acc.append(AE_acc)
        ah_acc.append(AH_acc)
        ac_acc.append(AC_acc)
        eg_acc.append(edge_acc)

    result = {
        'trans': tran_acc, 'EdgeChange': ae_acc, 'Edge': eg_acc,
        'HChange': ah_acc, 'ChargeChange': ac_acc
    }

    result = {k: torch.cat(v, dim=0).float() for k, v in result.items()}
    result = {k: v.mean().item() for k, v in result.items()}
    return result


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


def ablation_rc_train_trans(
    loader, model, optimizer, device, tokenizer, verbose=True, accu=1,
    warmup=False, pad='<PAD>', unk='<UNK>', label_smoothing=0.0,
    use_edge=False, use_ah=False, use_ac=False, use_ae=False
):
    model = model.train()

    results = {'trans': []}
    if use_edge:
        results['Edge'] = []
    if use_ah:
        results['HChange'] = []
    if use_ac:
        results['ChargeChange'] = []
    if use_ae:
        results['EdgeChange'] = []

    its, total_len = 1, len(loader)
    if warmup:
        warmup_iters = len(loader) - 1
        warmup_sher = warmup_lr_scheduler(optimizer, warmup_iters, 5e-2)

    ignore_idx = tokenizer.token2idx[pad]
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

        edge_logs, AH_logs, AE_logs, AC_logs, result = model(
            graphs=graphs, tgt=tgt_input, tgt_mask=sub_mask,
            tgt_pad_mask=pad_mask,
        )
        if use_ah:
            AH_loss = loss_batch(AH_logs, graphs.HChange, graphs.batch)
            results['HChange'].append(AH_loss.item())
        else:
            AH_loss = 0
        if use_ac:
            AC_loss = loss_batch(AC_logs, graphs.ChargeChange, graphs.batch)
            results['ChargeChange'].append(AC_loss.item())
        else:
            AC_loss = 0

        if use_ae:
            AE_loss = loss_batch(AE_logs, graphs.EdgeChange, graphs.batch)
            results['EdgeChange'].append(AE_loss.item())
        else:
            AE_loss = 0

        if use_edge:
            ed_loss = loss_batch(
                edge_logs, graphs.new_edge_types, graphs.e_batch
            )
            results['Edge'].append(ed_loss.item())
        else:
            ed_loss = 0

        loss_tran = calc_trans_loss(
            result, tgt_output, ignore_idx, label_smoothing
        )
        results['trans'].append(loss_tran.item())

        loss = AC_loss + AH_loss + AE_loss + ed_loss + loss_tran
        if not warmup and accu > 1:
            loss = loss / accu
        loss.backward()

        if its % accu == 0 or its == total_len or warmup:
            optimizer.step()
            optimizer.zero_grad()
        its += 1

        if warmup:
            warmup_sher.step()
    return {k: np.mean(v) for k, v in results.items()}


def ablation_rc_eval_trans(
    loader, model, device, tokenizer, pad='<PAD>', end='<END>',
    verbose=True, use_edge=False, use_ah=False, use_ac=False,
    use_ae=False
):
    model = model.eval()
    pad_idx = tokenizer.token2idx[pad]
    end_idx = tokenizer.token2idx[end]

    results = {'trans': []}
    if use_ae:
        results['EdgeChange'] = []
    if use_ah:
        results['HChange'] = []
    if use_ac:
        results['ChargeChange'] = []
    if use_edge:
        results['Edge'] = []

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
            edge_logs, AH_logs, AE_logs, AC_logs, result = model(
                graphs=graphs, tgt=tgt_input, tgt_mask=sub_mask,
                tgt_pad_mask=pad_mask,
            )

        if use_ae:
            AE_pred = convert_log_into_label(AE_logs, mod='softmax')
            AE_acc = eval_by_batch(
                AE_pred, graphs.EdgeChange,
                graphs.batch, return_tensor=True
            )
            results['EdgeChange'].append(AE_acc)

        if use_ah:
            AH_pred = convert_log_into_label(AH_logs, mod='softmax')
            AH_acc = eval_by_batch(
                AH_pred, graphs.HChange,
                graphs.batch, return_tensor=True
            )
            results['HChange'].append(AH_acc)

        if use_ac:
            AC_pred = convert_log_into_label(AC_logs, mod='softmax')
            AC_acc = eval_by_batch(
                AC_pred, graphs.ChargeChange,
                graphs.batch,  return_tensor=True
            )
            results['ChargeChange'].append(AC_acc)

        if use_edge:
            edge_pred = convert_edge_log_into_labels(
                edge_logs, graphs.edge_index,
                mod='softmax', return_dict=False
            )
            edge_acc = eval_by_batch(
                edge_pred, graphs.new_edge_types,
                graphs.e_batch, return_tensor=True
            )
            results['Edge'].append(edge_acc)

        trans_pred = convert_log_into_label(result, mod='softmax')
        trans_pred = correct_trans_output(trans_pred, end_idx, pad_idx)
        trans_acc = data_eval_trans(trans_pred, tgt_output, return_tensor=True)
        results['trans'].append(trans_acc)

    results = {k: torch.cat(v, dim=0).float() for k, v in results.items()}
    results = {k: v.mean().item() for k, v in results.items()}
    return result
