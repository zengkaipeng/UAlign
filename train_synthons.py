import torch
import argparse
import json
import os
import time
import pickle

from torch.utils.data import DataLoader
from training import train_synthon, eval_synthon
from data_utils import load_data, fix_seed, check_early_stop
from torch.optim.lr_scheduler import ExponentialLR
from sparse_backBone import GINBase, GATBase
from Mix_backbone import MixFormer
from model import EditDataset, SynthonModel, synthon_col_fn, make_edit_dataset


def create_log_model(args):
    timestamp = time.time()
    model_name = ('Gtrans_' if args.transformer else '') + args.gnn_type
    detail_log_folder = os.path.join(args.base_log, f'{model_name}')
    if not os.path.exists(detail_log_folder):
        os.makedirs(detail_log_folder)
    detail_log_dir = os.path.join(detail_log_folder, f'log-{timestamp}.json')
    detail_model_dir = os.path.join(detail_log_folder, f'mod-{timestamp}.pth')
    return detail_log_dir, detail_model_dir


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Graph Edit Exp, Sparse Model')
    parser.add_argument(
        '--dim', default=256, type=int,
        help='the hidden dim of model'
    )
    parser.add_argument(
        '--n_layer', default=8, type=int,
        help='the layer of encoder gnn'
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
        '--gnn_type', type=str, choices=['gat', 'gin'],
        help='type of gnn backbone', required=True
    )
    parser.add_argument(
        '--update_gate', choices=['add', 'cat'], type=str,
        help='the update gate for graphtransformer'
    )

    parser.add_argument(
        '--transformer', action='store_true',
        help='use graph transformer or not'
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
        '--device', default=-1, type=int,
        help='the device for running exps'
    )
    parser.add_argument(
        '--lr', default='1e-3', type=float,
        help='the learning rate for training'
    )
    parser.add_argument(
        '--base_log', default='log_synthons', type=str,
        help='the base dir of logging'
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
        '--use_class', action='store_true',
        help='use class for model or not'
    )
    parser.add_argument(
        '--num_workers', type=int, default=4,
        help='the number of num workers for dataloader'
    )

    args = parser.parse_args()
    print(args)
    log_dir, model_dir = create_log_model(args)

    if not torch.cuda.is_available() or args.device < 0:
        device = torch.device('cpu')
    else:
        device = torch.device(f'cuda:{args.device}')

    fix_seed(args.seed)

    train_rec, train_prod, train_rxn = load_data(args.data_path, 'train')
    val_rec, val_prod, val_rxn = load_data(args.data_path, 'val')
    test_rec, test_prod, test_rxn = load_data(args.data_path, 'test')

    print('[INFO] Data Loaded')

    train_set = make_edit_dataset(
        train_rec, train_prod,
        train_rxn if args.use_class else None
    )

    valid_set = make_edit_dataset(
        val_rec, val_prod,
        val_rxn if args.use_class else None
    )

    test_set = make_edit_dataset(
        test_rec, test_prod,
        test_rxn if args.use_class else None
    )

    train_loader = DataLoader(
        train_set, collate_fn=synthon_col_fn, batch_size=args.bs,
        shuffle=True, num_workers=args.num_workers
    )

    valid_loader = DataLoader(
        valid_set, collate_fn=synthon_col_fn, batch_size=args.bs,
        shuffle=False, num_workers=args.num_workers
    )
    test_loader = DataLoader(
        test_set, collate_fn=synthon_col_fn, batch_size=args.bs,
        shuffle=False, num_workers=args.num_workers
    )

    if args.transformer:
        if args.gnn_type == 'gin':
            gnn_args = {
                'in_channels': args.dim, 'out_channels': args.dim,
                'edge_dim': args.dim
            }
        elif args.gnn_type == 'gat':
            assert args.dim % args.heads == 0, \
                'The model dim should be evenly divided by num_heads'
            gnn_args = {
                'in_channels': args.dim, 'dropout': args.dropout,
                'out_channels': args.dim // args.heads, 'edge_dim': args.dim,
                'negative_slope': args.negative_slope, 'heads': args.heads
            }
        else:
            raise ValueError(f'Invalid GNN type {args.backbone}')

        GNN = MixFormer(
            emb_dim=args.dim, n_layers=args.n_layer, gnn_args=gnn_args,
            dropout=args.dropout, heads=args.heads, gnn_type=args.gnn_type,
            n_class=11 if args.use_class else None,
            update_gate=args.update_gate
        )
    else:
        if args.gnn_type == 'gin':
            GNN = GINBase(
                num_layers=args.n_layer, dropout=args.dropout,
                embedding_dim=args.dim,
                n_class=11 if args.use_class else None
            )
        elif args.gnn_type == 'gat':
            GNN = GATBase(
                num_layers=args.n_layer, dropout=args.dropout,
                embedding_dim=args.dim, num_heads=args.heads,
                negative_slope=args.negative_slope,
                n_class=11 if args.use_class else None
            )
        else:
            raise ValueError(f'Invalid GNN type {args.backbone}')

    model = SynthonModel(encoder=GNN, d_model=args.dim).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    lr_sh = ExponentialLR(optimizer, gamma=args.gamma, verbose=True)
    best_perf, best_ep = None, None

    log_info = {
        'args': args.__dict__, 'train_loss': [],
        'valid_metric': [], 'test_metric': []
    }

    with open(log_dir, 'w') as Fout:
        json.dump(log_info, Fout, indent=4)

    for ep in range(args.epoch):
        print(f'[INFO] traing at epoch {ep + 1}')
        train_loss = train_synthon(
            train_loader, model, optimizer, device, verbose=True,
            warmup=(ep < args.warmup), accu=args.accu,
        )
        log_info['train_loss'].append(train_loss)

        valid_result = eval_synthon(valid_loader, model, device, verbose=True)
        log_info['valid_metric'].append(valid_result)

        test_result = eval_synthon(test_loader, model, device, verbose=True)
        log_info['test_metric'].append(test_result)

        print('[TRAIN]', log_info['train_loss'][-1])
        print('[VALID]', log_info['valid_metric'][-1])
        print('[TEST]', log_info['test_metric'][-1])

        if ep >= args.warmup and ep >= args.step_start:
            lr_sh.step()

        with open(log_dir, 'w') as Fout:
            json.dump(log_info, Fout, indent=4)

        if best_perf is None or valid_result['Edge'] > best_perf:
            best_perf, best_ep = valid_result['Edge'], ep
            torch.save(model.state_dict(), model_dir)

        if args.early_stop > 3 and ep > max(10, args.early_stop):
            tx = log_info['valid_metric'][-args.early_stop:]
            tx = [x['Edge'] for x in tx]
            if check_early_stop(tx):
                break

    print(f'[INFO] best acc epoch: {best_ep}')
    print(f'[INFO] best valid loss: {log_info["valid_metric"][best_ep]}')
    print(f'[INFO] best test loss: {log_info["test_metric"][best_ep]}')