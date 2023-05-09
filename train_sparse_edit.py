import torch
import argparse
import json
import os

from torch.utils.data import DataLoader
from sparse_backBone import GINBase, GATBase, sparse_edit_collect_fn
from model import GraphEditModel
from training import train_sparse_edit, eval_sparse_edit
from data_utils import create_sparse_dataset, load_data, fix_seed

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

    args = parser.parse_args()

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

    train_loader = DataLoader(
        train_set, collate_fn=sparse_edit_collect_fn,
        batch_size=args.bs, shuffle=True
    )
    valid_loader = DataLoader(
        valid_set, collate_fn=sparse_edit_collect_fn,
        batch_size=args.bs, shuffle=False
    )
    test_loader = DataLoader(
        test_set, collate_fn=sparse_edit_collect_fn,
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
            num_heads=args.heads
        )

    model = GraphEditModel(
        GNN, True, embedding_dim, embedding_dim,
        4 if args.kekulize else 5
    )

    for ep in range(args.epoch):
        print(f'[INFO] traing at epoch {ep + 1}')
        train_sparse_edit(loader, model, optimizer)
