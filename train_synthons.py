import torch
import argparse
import json
import os
import time

from torch.utils.data import DataLoader
from sparse_backBone import GINBase, GATBase
from Mix_backbone import MixFormer
from Dataset import edit_col_fn
from model import SynthonPredictionModel
from training import train_sparse_edit, eval_sparse_edit
from data_utils import (
    create_edit_dataset, load_data, fix_seed,
    check_early_stop
)


def create_log_model(args):
    timestamp = time.time()
    detail_log_folder = os.path.join(
        args.base_log, 'with_class' if args.use_class else 'wo_class',
        ('Gtrans_' if args.transformer else '') + args.gnn_type
    )
    if not os.path.exists(detail_log_folder):
        os.makedirs(detail_log_folder)
    detail_log_dir = os.path.join(detail_log_folder, f'log-{timestamp}.json')
    detail_model_dir = os.path.join(detail_log_folder, f'mod-{timestamp}.pth')
    return detail_log_dir, detail_model_dir


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Graph Edit Exp, Sparse Model')
    # public setting
    parser.add_argument(
        '--dim', default=256, type=int,
        help='the hidden dim of model'
    )
    parser.add_argument(
        '--kekulize', action='store_true',
        help='kekulize molecules if it\'s added'
    )
    parser.add_argument(
        '--n_layer', default=5, type=int,
        help='the layer of backbones'
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
        '--gnn_type', type=str, choices=['gat', 'gin'],
        help='type of gnn backbone', required=True
    )
    parser.add_argument(
        '--dropout', type=float, default=0.1,
        help='the dropout rate, useful for all backbone'
    )

    parser.add_argument(
        '--base_log', default='log_edit', type=str,
        help='the base dir of logging'
    )

    # GAT & Gtrans setting
    parser.add_argument(
        '--transformer', action='store_true',
        help='use graph transformer or not'
    )
    parser.add_argument(
        '--heads', default=4, type=int,
        help='the number of heads for attention, only useful for gat'
    )
    parser.add_argument(
        '--negative_slope', type=float, default=0.2,
        help='negative slope for attention, only useful for gat'
    )
    parser.add_argument(
        '--update_gate', choices=['cat', 'add'], default='add',
        help='the update method for mixformer', type=str,
    )

    # training
    parser.add_argument(
        '--pos_weight', default=1, type=float,
        help='the weight for positive samples'
    )

    args = parser.parse_args()
    print(args)

    log_dir, model_dir, fit_dir = create_log_model(args)

    if not torch.cuda.is_available() or args.device < 0:
        device = torch.device('cpu')
    else:
        device = torch.device(f'cuda:{args.device}')

    fix_seed(args.seed)

    train_rec, train_prod, train_rxn = load_data(args.data_path, 'train')
    val_rec, val_prod, val_rxn = load_data(args.data_path, 'val')
    test_rec, test_prod, test_rxn = load_data(args.data_path, 'test')

    train_set = create_edit_dataset(
        reacts=train_rec, prods=train_prod, kekulize=args.kekulize,
        rxn_class=train_rxn if args.use_class else None,
    )

    valid_set = create_edit_dataset(
        reacts=val_rec, prods=val_prod, kekulize=args.kekulize,
        rxn_class=val_rxn if args.use_class else None,
    )
    test_set = create_edit_dataset(
        reacts=test_rec, prods=test_prod, kekulize=args.kekulize,
        rxn_class=test_rxn if args.use_class else None,
    )

    train_loader = DataLoader(
        train_set, collate_fn=edit_col_fn,
        batch_size=args.bs, shuffle=True
    )
    valid_loader = DataLoader(
        valid_set, collate_fn=edit_col_fn,
        batch_size=args.bs, shuffle=False
    )
    test_loader = DataLoader(
        test_set, collate_fn=edit_col_fn,
        batch_size=args.bs, shuffle=False
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
                embedding_dim=args.dim, n_class=11 if args.use_class else None
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

    model = BinaryGraphEditModel(GNN, args.dim, args.dim, args.dropout)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_node, node_ep, best_edge, edge_ep = [None] * 4

    log_info = {
        'args': args.__dict__, 'train_loss': [],
        'valid_metric': [], 'test_metric': []
    }

    with open(log_dir, 'w') as Fout:
        json.dump(log_info, Fout, indent=4)

    for ep in range(args.epoch):
        print(f'[INFO] traing at epoch {ep + 1}')
        node_loss, edge_loss = train_sparse_edit(
            train_loader, model, optimizer, device, verbose=True,
            warmup=(ep == 0),  pos_weight=args.pos_weight
        )
        log_info['train_loss'].append({'node': node_loss, 'edge': edge_loss})

        print('[TRAIN]', log_info['train_loss'][-1])
        valid_results = eval_sparse_edit(valid_loader, model, device, True)
        log_info['valid_metric'].append(valid_results)

        print('[VALID]', log_info['valid_metric'][-1])

        test_results = eval_sparse_edit(test_loader, model, device, True)
        log_info['test_metric'].append(test_results)

        print('[TEST]', log_info['test_metric'][-1])

        with open(log_dir, 'w') as Fout:
            json.dump(log_info, Fout, indent=4)

        if best_node is None or valid_results['by_node']['fit'] > best_node:
            best_node, node_ep = valid_results['by_node']['fit'], ep
            torch.save(model.state_dict(), model_dir)
        if best_edge is None or valid_results['by_edge']['fit'] > best_edge:
            best_edge, edge_ep = valid_results['by_edge']['fit'], ep
            torch.save(model.state_dict(), fit_dir)
        if args.early_stop > 5 and ep > max(20, args.early_stop):
            val_his = log_info['valid_metric'][-args.early_stop:]
            nf = [x['by_node']['fit'] for x in val_his]
            ef = [x['by_edge']['fit'] for x in val_his]

            if check_early_stop(nf, ef):
                break
