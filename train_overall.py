import torch
import argparse
import json
import os
import time

from torch.utils.data import DataLoader
from sparse_backBone import GINBase, GATBase, GCNBase
from Mix_backbone import MixFormer
from Dataset import edit_col_fn
from model import BinaryGraphEditModel
from data_utils import (
    create_overall_dataset, load_data, fix_seed,
    check_early_stop
)
from utils.chemistry_parse import canonical_smiles
from copy import deepcopy
from decoder import MixDecoder, GINDecoder, GATDecoder, GCNDecoder


def create_log_model(args):
    if args.pos_enc == 'Lap' and args.lap_pos_dim <= 0:
        raise ValueError('The dim of positional enc should be positive')

    timestamp = time.time()
    detail_log_folder = os.path.join(
        args.base_log, 'with_class' if args.use_class else 'wo_class',
        ('Gtrans_' if args.transformer else '') + args.gnn_type
    )
    if not os.path.exists(detail_log_folder):
        os.makedirs(detail_log_folder)
    detail_log_dir = os.path.join(detail_log_folder, f'log-{timestamp}.json')
    model_dir = os.path.join(detail_log_folder, f'model-{timestamp}.pth')
    return detail_log_dir, model_dir


if __name__ == '__main__':
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
        '--gnn_type', type=str, choices=['gat', 'gin', 'gcn'],
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
        '--pos_enc', choices=['none', 'Lap'], type=str, default='none',
        help='the method to add graph positional encoding'
    )

    parser.add_argument(
        '--lap_pos_dim', type=int, default=5,
        help='the dim of lap pos encoding'
    )
    parser.add_argument(
        '--update_gate', choices=['cat', 'add', 'gate'], default='add',
        help='the update method for mixformer', type=str,
    )

    # training
    parser.add_argument(
        '--mode', choices=['all', 'org', 'merge'], type=str,
        help='the training mode for synthon prediction',
        default='all'
    )
    parser.add_argument(
        '--reduction', choices=['sum', 'mean'], type=str,
        default='mean', help='the method to reduce loss'
    )
    parser.add_argument(
        '--graph_level', action='store_true',
        help='calc loss in graph level'
    )
    parser.add_argument(
        '--pad_num', type=int, default=35, 
        help='the number of padding'
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

    if args.pos_enc == 'none':
        dataset_kwargs = {'pos_enc': args.pos_enc}
    elif args.pos_enc == 'Lap':
        dataset_kwargs = {'pos_enc': args.pos_enc, 'dim': args.lap_pos_dim}
    else:
        raise ValueError(f'Invalid pos_enc {args.pos_enc}')

    train_set = create_overall_dataset(
        reacts=train_rec, prods=train_prod, kekulize=args.kekulize,
        rxn_class=train_rxn if args.use_class else None, **dataset_kwargs
    )

    valid_set = create_overall_dataset(
        reacts=val_rec, prods=val_prod, kekulize=args.kekulize,
        rxn_class=val_rxn if args.use_class else None, **dataset_kwargs
    )
    test_set = create_overall_dataset(
        reacts=test_rec, prods=test_prod, kekulize=args.kekulize,
        rxn_class=test_rxn if args.use_class else None, **dataset_kwargs
    )

    val_node_types = valid_set.decoder_node_class
    test_node_types = test_set.decoder_node_class

    val_edge_types = valid_set.decoder_edge_class
    test_node_types = test_set.decoder_edge_class

    val_rec_smi = [canonical_smiles(x) for x in val_rec]
    test_rec_smi = [canonical_smiles(x) for x in test_rec]


    if args.transformer:
        if args.pos_enc == 'Lap':
            pos_args = {'dim': args.lap_pos_dim}
        else:
            pos_args = None

        if args.gnn_type == 'gcn':
            gnn_args = {'emb_dim': args.dim}
        elif args.gnn_type == 'gin':
            gnn_args = {'embedding_dim': args.dim}
        elif args.gnn_type == 'gat':
            assert args.dim % args.heads == 0, \
                'The model dim should be evenly divided by num_heads'
            gnn_args = {
                'in_channels': args.dim,
                'out_channels': args.dim // args.heads,
                'negative_slope': args.negative_slope,
                'dropout': args.dropout, 'add_self_loop': False,
                'edge_dim': args.dim, 'heads': args.heads
            }
        else:
            raise ValueError(f'Invalid GNN type {args.backbone}')

        GNN = MixFormer(
            emb_dim=args.dim, n_layers=args.n_layer, gnn_args=gnn_args,
            dropout=args.dropout, heads=args.heads, pos_enc=args.pos_enc,
            negative_slope=args.negative_slope, pos_args=pos_args,
            n_class=11 if args.use_class else None, edge_last=True,
            residual=True, update_gate=args.update_gate, gnn_type=args.gnn_type
        )
        GNN_dec = MixDecoder(
            emb_dim=args.dim, n_layers=args.n_layer, gnn_args=gnn_args, 
            n_pad=args.pad_num
        )
    else:
        if args.gnn_type == 'gin':
            GNN = GINBase(
                num_layers=args.n_layer, dropout=args.dropout, residual=True,
                embedding_dim=args.dim, edge_last=True,
                n_class=11 if args.use_class else None
            )
        elif args.gnn_type == 'gat':
            GNN = GATBase(
                num_layers=args.n_layer, dropout=args.dropout, self_loop=False,
                embedding_dim=args.dim, edge_last=True, residual=True,
                negative_slope=args.negative_slope, num_heads=args.heads,
                n_class=11 if args.use_class else None
            )
        elif args.gnn_type == 'gcn':
            GNN = GCNBase(
                num_layers=args.n_layer, dropout=args.dropout, residual=True,
                embedding_dim=args.dim, edge_last=True,
                n_class=11 if args.use_class else None
            )
        else:
            raise ValueError(f'Invalid GNN type {args.backbone}')

    encoder = BinaryGraphEditModel(GNN, args.dim, args.dim, args.dropout)



