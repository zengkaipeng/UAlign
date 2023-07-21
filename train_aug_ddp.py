import torch
import argparse
import json
import os
import time
import pickle


from tokenlizer import DEFAULT_SP, Tokenizer
from torch.utils.data import DataLoader
from sparse_backBone import GINBase, GATBase
from model import Graph2Seq, fc_collect_fn, PositionalEncoding, Acc_fn
from training import train_trans, eval_trans
from data_utils import create_sparse_dataset, load_data, fix_seed
from torch.nn import TransformerDecoderLayer, TransformerDecoder
from torch.optim.lr_scheduler import ExponentialLR
import torch.distributed as torch_dist
import torch.multiprocessing as torch_mp
from torch.utils.data.distributed import DistributedSampler
from ddp_training import ddp_train_trans, ddp_eval_trans


def create_log_model(args):
    timestamp = time.time()
    log_dir = [
        f'dim_{args.dim}', f'seed_{args.seed}', f'dropout_{args.dropout}',
        f'bs_{args.bs}', f'lr_{args.lr}', f'heads_{args.heads}',
        f'encoder_{args.layer_encoder}', f'decoder_{args.layer_decoder}',
        f'label_smooth_{args.label_smooth}', f'warm_{args.warmup}',
        f'accu_{args.accu}', f'gamma_{args.gamma}',
        f'lrstep_{args.step_start}', f'aug_prob_{args.aug_prob}'
    ]
    if args.kekulize:
        log_dir.append('kekulize')

    detail_log_folder = os.path.join(
        args.base_log,
        'with_class' if args.use_class else 'wo_class',
        args.backbone, '-'.join(log_dir)
    )
    if not os.path.exists(detail_log_folder):
        os.makedirs(detail_log_folder)
    detail_log_dir = os.path.join(detail_log_folder, f'log-{timestamp}.json')
    detail_model_dir = os.path.join(detail_log_folder, f'mod-{timestamp}.pth')
    token_dir = os.path.join(detail_log_folder, f'token-{timestamp}.pkl')
    bacc_dir = os.path.join(detail_log_folder, f'acc-{timestamp}.pth')
    return detail_log_dir, detail_model_dir, token_dir, bacc_dir


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Graph Edit Exp, Sparse Model')
    parser.add_argument(
        '--dim', default=256, type=int,
        help='the hidden dim of model'
    )
    parser.add_argument(
        '--port', type=str, default='12345',
        help='the port for ddp message passing'
    )
    parser.add_argument(
        '--num_gpus', type=int, required=True,
        help='the number of used gpus'
    )
    parser.add_argument(
        '--kekulize', action='store_true',
        help='kekulize molecules if it\'s added'
    )
    parser.add_argument(
        '--aug_prob', default=0.5, type=float,
        help='the probability of performing data augumentation '
        "should be between 0 and 1"
    )
    parser.add_argument(
        '--layer_encoder', default=8, type=int,
        help='the layer of encoder gnn'
    )
    parser.add_argument(
        '--layer_decoder', default=8, type=int,
        help='the layer of transformer decoder'
    )
    parser.add_argument(
        '--token_path', required=True, type=str,
        help='the path of a json containing all tokens'
    )
    parser.add_argument(
        '--heads', default=4, type=int,
        help='the number of heads for attention, only useful for gat'
    )
    parser.add_argument(
        '--warmup', default=1, type=int,
        help='the epoch of warmup'
    )
    parser.add_argument(
        '--backbone', type=str, choices=['GAT', 'GIN'],
        help='type of gnn backbone', required=True
    )
    parser.add_argument(
        '--gamma', default=0.998, type=float,
        help='the gamma of lr scheduler'
    )
    parser.add_argument(
        '--dropout', type=float, default=0.3,
        help='the dropout rate, useful for all backbone'
    )
    parser.add_argument(
        '--negative_slope', type=float, default=0.2,
        help='negative slope for attention, only useful for gat'
    )
    parser.add_argument(
        '--data_path', required=True, type=str,
        help='the path containing dataset'
    )
    parser.add_argument(
        '--use_class', action='store_true',
        help='use rxn_class for training or not'
    )
    parser.add_argument(
        '--seed', type=int, default=2023,
        help='the seed for training'
    )
    parser.add_argument(
        '--bs', type=int, default=512,
        help='the batch size for training'
    )
    parser.add_argument(
        '--epoch', type=int, default=200,
        help='the max epoch for training'
    )
    parser.add_argument(
        '--early_stop', default=0, type=int,
        help='number of epochs to judger early stop '
        ', will be ignored when it\'s less than 5'
    )
    parser.add_argument(
        '--lr', default='1e-3', type=float,
        help='the learning rate for training'
    )
    parser.add_argument(
        '--base_log', default='log_exp', type=str,
        help='the base dir of logging'
    )
    parser.add_argument(
        '--label_smooth', default=0.0, type=float,
        help='the label smoothing for transformer'
    )
    parser.add_argument(
        '--accu', type=int, default=1,
        help='the number of batch accu'
    )
    parser.add_argument(
        '--step_start', type=int, default=50,
        help='the step of starting lr decay'
    )
    parser.add_argument(
        '--checkpoint', type=str, default='',
        help='the path of checkpoint to restart the exp'
    )
    parser.add_argument(
        '--token_ckpt', type=str, default='',
        help='the path of tokenizer, when ckpt is loaded, necessary'
    )

    args = parser.parse_args()
    print(args)
    log_dir, model_dir, token_dir, acc_dir = create_log_model(args)

    fix_seed(args.seed)
    torch_mp.spawn(main_worker, args=(args, ))


def main_worker(worker_idx, args):
    torch_dist.init_process_group(
        backend='nccl', init_method=f'tcp://127.0.0.1:{args.port}',
        world_size=args.num_gpus, rank=worker_idx
    )

    torch.cuda.set_device(worker_idx)

    with open(args.token_path) as Fin:
        ALL_TOKEN = json.load(Fin)

    if args.use_class:
        SP_TOKEN = DEFAULT_SP | {f'<RXN_{i}>' for i in range(10)}
    else:
        SP_TOKEN = DEFAULT_SP

    tokenizer = Tokenizer(ALL_TOKEN, SP_TOKEN)

    train_rec, train_prod, train_rxn = load_data(args.data_path, 'train')
    val_rec, val_prod, val_rxn = load_data(args.data_path, 'val')
    test_rec, test_prod, test_rxn = load_data(args.data_path, 'test')

    print(f'[INFO {worker_idx}] Data Loaded')

    train_set = create_sparse_dataset(
        train_rec, train_prod, kekulize=args.kekulize,
        rxn_class=train_rxn if args.use_class else None,
        randomize=True, aug_prob=args.aug_prob
    )
    valid_set = create_sparse_dataset(
        val_rec, val_prod, kekulize=args.kekulize,
        rxn_class=val_rxn if args.use_class else None,
        randomize=False
    )
    test_set = create_sparse_dataset(
        test_rec, test_prod, kekulize=args.kekulize,
        rxn_class=test_rxn if args.use_class else None,
        randomize=False
    )

    train_sampler = DistributedSampler(train_set, shuffle=True)
    valid_sampler = DistributedSampler(valid_set, shuffle=False)
    test_sampler = DistributedSampler(test_set, shuffle=False)

    train_loader = DataLoader(
        train_set, collate_fn=fc_collect_fn, batch_size=args.bs,
        shuffle=False, batch_sampler=train_sampler, pin_memory=True
    )
    valid_loader = DataLoader(
        valid_set, collate_fn=fc_collect_fn, batch_size=args.bs,
        shuffle=False, batch_sampler=valid_sampler, pin_memory=True,
    )
    test_loader = DataLoader(
        test_set, collate_fn=fc_collect_fn, batch_size=args.bs,
        shuffle=False, batch_sampler=test_sampler, pin_memory=True
    )

    if args.backbone == 'GIN':
        GNN = GINBase(
            num_layers=args.layer_encoder, dropout=args.dropout,
            embedding_dim=args.dim, edge_last=True, residual=True
        )
    else:
        GNN = GATBase(
            num_layers=args.layer_encoder, dropout=args.dropout,
            embedding_dim=args.dim, edge_last=True,
            residual=True, negative_slope=args.negative_slope,
            num_heads=args.heads, add_self_loop=True
        )

    decode_layer = TransformerDecoderLayer(
        d_model=args.dim, nhead=args.heads, batch_first=True,
        dim_feedforward=args.dim * 2, dropout=args.dropout
    )
    Decoder = TransformerDecoder(decode_layer, args.layer_decoder)
    Pos_env = PositionalEncoding(args.dim, args.dropout, maxlen=2000)

    model = Graph2Seq(
        token_size=tokenizer.get_token_size(), encoder=GNN,
        decoder=Decoder, d_model=args.dim, pos_enc=Pos_env
    ).cuda()

    if args.checkpoint != '':
        assert args.token_ckpt != '', 'Missing Tokenizer Information'
        print(f'[INFO {worker_idx}] Loading model weight in {args.checkpoint}')
        weight = torch.load(args.checkpoint, map_location=torch.device('cuda'))
        model.load_state_dict(weight)

    if args.token_ckpt != '':
        print(f'[INFO {worker_idx}] Loading tokenizer from {args.token_ckpt}')
        with open(args.token_ckpt, 'rb') as Fin:
            tokenizer = pickle.load(Fin)

    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[worker_idx]
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    verbose = (worker_idx == 0)
    lr_sh = ExponentialLR(optimizer, gamma=args.gamma, verbose=verbose)
    best_perf, best_ep = None, None
    best_acc, best_ep2 = None, None

    node_fn = torch.nn.CrossEntropyLoss(reduction='sum')
    edge_fn = torch.nn.CrossEntropyLoss(reduction='sum')
    tran_fn = torch.nn.CrossEntropyLoss(
        reduction='sum', ignore_index=tokenizer.token2idx['<PAD>'],
        label_smoothing=args.label_smooth
    )
    valid_fn = torch.nn.CrossEntropyLoss(
        reduction='sum', ignore_index=tokenizer.token2idx['<PAD>']
    )

    acc_fn = Acc_fn(ignore_index=tokenizer.token2idx['<PAD>'])
    print(f'[INFO {worker_idx}] padding index', tokenizer.token2idx['<PAD>'])

    log_info = {
        'args': args.__dict__, 'train_loss': [],
        'valid_metric': [], 'test_metric': []
    }

    if verbose:
        with open(token_dir, 'wb') as Fout:
            pickle.dump(tokenizer, Fout)

        with open(log_dir, 'w') as Fout:
            json.dump(log_info, Fout, indent=4)

    for ep in range(args.epoch):
        if verbose:
            print(f'[INFO] traing at epoch {ep + 1}')

        train_sampler.set_epoch(ep)
        train_metrics = ddp_train_trans(
            train_loader, model, optimizer, tokenizer, node_fn,
            edge_fn, tran_fn, acc_fn, verbose=verbose,
            warmup=(ep < args.warmup), accu=args.accu
        )
        val_metrics = ddp_eval_trans(
            valid_loader, model, valid_fn, tokenizer,
            acc_fn, verbose=verbose
        )
        test_metrics = ddp_eval_trans(
            test_loader, model, valid_fn, tokenizer,
            acc_fn, verbose=verbose
        )
        torch_dist.barrier()

        train_metrics.all_reduct()
        val_metrics.all_reduct()
        test_metrics.all_reduct()

        log_info['train_loss'].append(train_metrics.get_all_value_dict())
        log_info['valid_metric'].append(val_metrics.get_all_value_dict())
        log_info['test_metric'].append(test_metrics.get_all_value_dict())

        if verbose:
            print('[TRAIN]', log_info['train_loss'][-1])
            print('[VALID]', log_info['valid_metric'][-1])
            print('[TEST]', log_info['test_metric'][-1])

            with open(log_dir, 'w') as Fout:
                json.dump(log_info, Fout, indent=4)

            valid_results = log_info['valid_metric'][-1]['tloss']
            valid_acc = log_info['valid_metric'][-1]['acc']
            if best_perf is None or valid_results < best_perf:
                best_perf, best_ep = valid_results, ep
                torch.save(model.module.state_dict(), model_dir)

            if best_acc is None or valid_acc > best_acc:
                best_acc, best_ep2 = valid_acc, ep
                torch.save(model.module.state_dict(), acc_dir)

        if ep >= args.warmup and ep >= args.step_start:
            lr_sh.step()

        if args.early_stop > 3 and ep > args.early_stop:
            tx = [
                x['tloss'] for x in
                log_info['valid_metric'][-args.early_stop:]
            ]
            ty = [
                x['acc'] for x in
                log_info['valid_metric'][-args.early_stop:]
            ]
            if all(x >= tx[0] for x in tx) and all(x <= ty[0] for x in ty):
                print(f'[INFO {worker_idx}] early_stop_break')
                break
    if not verbose:
        return

    print(f'[INFO] best loss epoch: {best_ep}')
    print(f'[INFO] best valid loss: {log_info["valid_metric"][best_ep]}')
    print(f'[INFO] best test loss: {log_info["test_metric"][best_ep]}')

    print(f'[INFO] best acc epoch: {best_ep2}')
    print(f'[INFO] best valid loss: {log_info["valid_metric"][best_ep2]}')
    print(f'[INFO] best test loss: {log_info["test_metric"][best_ep2]}')
