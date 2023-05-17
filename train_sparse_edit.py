import torch
import argparse
import json
import os
import time

from torch.utils.data import DataLoader
from sparse_backBone import GINBase, GATBase
from model import GraphEditModel, get_collate_fn
from training import train_sparse_edit, eval_sparse_edit
from data_utils import (
    create_sparse_dataset, load_data, fix_seed,
    check_early_stop
)


def create_log_model(args):
    timestamp = time.time()
    log_dir = [
        f'dim_{args.dim}', f'n_layer_{args.n_layer}', f'seed_{args.seed}'
        f'dropout_{args.dropout}', f'bs_{args.bs}', f'lr_{args.lr}',
        f'mode_{args.mode}', 'kekulize' if args.kekulize else ''
    ]
    if args.backbone == 'GAT' and args.add_self_loop:
        log_dir.append('self_loop')

    detail_log_folder = os.path.join(
        args.base_log,
        'with_class' if args.use_class else 'wo_class',
        args.backbone, '-'.join(log_dir)
    )
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
        '--kekulize', action='store_true',
        help='kekulize molecules if it\'s added'
    )
    parser.add_argument(
        '--n_layer', default=4, type=int,
        help='the layer of backbones'
    )
    parser.add_argument(
        '--heads', default=4, type=int,
        help='the number of heads for attention, only useful for gat'
    )
    parser.add_argument(
        '--backbone', type=str, choices=['GAT', 'GIN'],
        help='type of gnn backbone', required=True
    )
    parser.add_argument(
        '--dropout', type=float, default=0.1,
        help='the dropout rate, useful for all backbone'
    )
    parser.add_argument(
        '--negative_slope', type=float, default=0.2,
        help='negative slope for attention, only useful for gat'
    )
    parser.add_argument(
        '--mode', choices=['together', 'original'], type=str,
        help='the training mode, together will extend the '
        'training data using the node result'
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
        '--early_stop', default=10, type=int,
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
        '--base_log', default='log', type=str,
        help='the base dir of logging'
    )
    parser.add_argument(
        '--add_self_loop', action='store_true',
        help='explictly add self loop in the graph data'
        ' only useful for gat'
    )

    args = parser.parse_args()
    if args.backbone == 'GIN':
        args.add_self_loop = False
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

    train_set = create_sparse_dataset(
        train_rec, train_prod, kekulize=args.kekulize,
        rxn_class=train_rxn if args.use_class else None
    )

    valid_set = create_sparse_dataset(
        val_rec, val_prod, kekulize=args.kekulize,
        rxn_class=val_rxn if args.use_class else None
    )
    test_set = create_sparse_dataset(
        test_rec, test_prod, kekulize=args.kekulize,
        rxn_class=test_rxn if args.use_class else None
    )

    col_fn = get_collate_fn(sparse=True, self_loop=args.add_self_loop)

    train_loader = DataLoader(
        train_set, collate_fn=col_fn,
        batch_size=args.bs, shuffle=True
    )
    valid_loader = DataLoader(
        valid_set, collate_fn=col_fn,
        batch_size=args.bs, shuffle=False
    )
    test_loader = DataLoader(
        test_set, collate_fn=col_fn,
        batch_size=args.bs, shuffle=False
    )

    if args.backbone == 'GIN':
        GNN = GINBase(
            num_layers=args.n_layer, dropout=args.dropout,
            embedding_dim=args.dim, edge_last=False, residual=True
        )
    else:
        GNN = GATBase(
            num_layers=args.n_layer, dropout=args.dropout,
            embedding_dim=args.dim, edge_last=False,
            residual=True, negative_slope=args.negative_slope,
            num_heads=args.heads, self_loop=not args.add_self_loop
        )

    model = GraphEditModel(
        GNN, True, args.dim, args.dim,
        4 if args.kekulize else 5
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_perf, best_ep = None, None

    log_info = {
        'args': args.__dict__, 'train_loss': [],
        'valid_metric': [], 'test_metric': []
    }

    with open(log_dir, 'w') as Fout:
        json.dump(log_info, Fout, indent=4)

    for ep in range(args.epoch):
        print(f'[INFO] traing at epoch {ep + 1}')
        node_loss, edge_loss = train_sparse_edit(
            train_loader, model, optimizer, device,
            verbose=True, warmup=(ep == 0), mode=args.mode
        )
        log_info['train_loss'].append({
            'node': node_loss, 'edge': edge_loss
        })
        valid_results = eval_sparse_edit(
            valid_loader, model, device, verbose=True
        )
        log_info['valid_metric'].append({
            'node_cover': valid_results[0], 'node_fit': valid_results[1],
            'edge_fit': valid_results[2], 'all_cover': valid_results[3],
            'all_fit': valid_results[4]
        })

        test_results = eval_sparse_edit(
            test_loader, model, device, verbose=True
        )

        log_info['test_metric'].append({
            'node_cover': test_results[0], 'node_fit': test_results[1],
            'edge_fit': test_results[2], 'all_cover': test_results[3],
            'all_fit': test_results[4]
        })

        with open(log_dir, 'w') as Fout:
            json.dump(log_info, Fout, indent=4)
        if best_perf is None or valid_results[3] > best_perf:
            best_perf, best_ep = valid_results[3], ep
            torch.save(model.state_dict(), model_dir)

        if args.early_stop > 5 and ep > max(20, args.early_stop):
            nc = [
                x['node_cover'] for x in
                log_info['valid_metric'][-args.early_stop:]
            ]
            ef = [
                x['edge_fit'] for x in
                log_info['valid_metric'][-args.early_stop:]
            ]
            if check_early_stop(nc, ef):
                break
