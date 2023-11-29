import torch
import argparse
import json
import os
import time

from torch.utils.data import DataLoader
from sparse_backBone import GINBase, GATBase
from Mix_backbone import MixFormer
from Dataset import overall_col_fn
from model import OverallModel
from data_utils import (
    create_overall_dataset, load_data,
    fix_seed, check_early_stop
)
from training import train_overall, eval_overall
from torch.optim.lr_scheduler import ExponentialLR


def create_log_model(args):

    timestamp = time.time()
    detail_log_folder = os.path.join(
        args.base_log, 'with_class' if args.use_class else 'wo_class',
        ('Gtrans_' if args.transformer else '') + args.gnn_type
    )
    if not os.path.exists(detail_log_folder):
        os.makedirs(detail_log_folder)
    detail_log_dir = os.path.join(detail_log_folder, f'log-{timestamp}.json')
    model_dir = os.path.join(detail_log_folder, f'loss-{timestamp}.pth')
    acc_dir = os.path.join(detail_log_folder, f'acc-{timestamp}.pth')
    return detail_log_dir, model_dir, acc_dir


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
        '--base_log', default='log_overall', type=str,
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
    parser.add_argument(
        '--warmup', default=2, type=int,
        help='the num of epoch for warmup'
    )

    parser.add_argument(
        '--lrgamma', default=1, type=float,
        help='the gamma for lr step'
    )
    parser.add_argument(
        '--checkpoint', default='', type=str,
        help='the path of trained overall model, to restart the exp'
    )
    parser.add_argument(
        '--pregraph', action='store_true',
        help='cat the graph embedding before the transformer ' +
        'encoder, if not chosen, cat it after transformer encoder'
    )
    parser.add_argument(
        '--use_sim', action='store_true',
        help='use the sim model while making link prediction ' +
        'between leaving groups and synthons'
    )
    parser.add_argument(
        '--token_path', type=str, required=True,
        help='the json file containing all tokens'
    )
    parser.add_argument(
        '--token_ckpt', type=str, default='',
        help='the path of token checkpoint, required while' +
        ' checkpoint is specified'
    )

    args = parser.parse_args()
    print(args)

    log_dir, model_dir, acc_dir = create_log_model(args)

    if not torch.cuda.is_available() or args.device < 0:
        device = torch.device('cpu')
    else:
        device = torch.device(f'cuda:{args.device}')

    fix_seed(args.seed)

    train_rec, train_prod, train_rxn = load_data(args.data_path, 'train')
    val_rec, val_prod, val_rxn = load_data(args.data_path, 'val')
    test_rec, test_prod, test_rxn = load_data(args.data_path, 'test')

    train_set = create_overall_dataset(
        reacts=train_rec, prods=train_prod, kekulize=args.kekulize,
        rxn_class=train_rxn if args.use_class else None, verbose=True
    )
    valid_set = create_overall_dataset(
        reacts=val_rec, prods=val_prod, kekulize=args.kekulize,
        rxn_class=val_rxn if args.use_class else None, verbose=True
    )
    test_set = create_overall_dataset(
        reacts=test_rec, prods=test_prod, kekulize=args.kekulize,
        rxn_class=val_rxn if args.use_class else None, verbose=True
    )

    train_loader = DataLoader(
        train_set, collate_fn=overall_col_fn,
        batch_size=args.bs, shuffle=True
    )
    valid_loader = DataLoader(
        valid_set, collate_fn=overall_col_fn,
        batch_size=args.bs, shuffle=False
    )
    test_loader = DataLoader(
        test_set, collate_fn=overall_col_fn,
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

    enc_layer = torch.nn.TransformerEncoderLayer(
        args.dim, args.heads, dim_feedforward=args.dim * 2,
        batch_first=True, dropout=args.dropout
    )
    dec_layer = torch.nn.TransformerDecoderLayer(
        args.dim, args.heads, dim_feedforward=args.dim * 2,
        batch_first=True, dropout=args.dropout
    )
    TransEnc = torch.nn.TransformerEncoder(enc_layer, args.n_layer)
    TransDec = torch.nn.TransformerDecoder(dec_layer, args.n_layer)

    model = OverallModel(
        GNN, TransEnc, TransDec, args.dim, args.dim, num_token,
        heads=args.heads, dropout=args.dropout, use_sim=args.use_sim,
        pre_graph=args.pregraph, rxn_num=11 if args.use_class else None
    )

    model = EncoderDecoder(encoder, decoder).to(device)

    if args.checkpoint != '':
        weight = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(weight)
    elif args.encoder_ckpt != '':
        weight = torch.load(args.encoder_ckpt, map_location=device)
        model.encoder.load_state_dict(weight, strict=False)

    print('[INFO] model built')

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scher = ExponentialLR(optimizer, args.lrgamma, verbose=True)
    best_perf, best_epoch = None, None
    log_info = {
        'args': args.__dict__, 'train_loss': [],
        'valid_metric': [], 'test_metric': []
    }

    with open(log_dir, 'w') as Fout:
        json.dump(log_info, Fout, indent=4)

    for ep in range(args.epoch):
        print(f'[INFO] traning for ep {ep}')
        train_loss = train_overall(
            model, train_loader, optimizer, device, pos_weight=args.pos_weight,
            alpha=args.alpha, matching=args.matching, warmup=ep < args.warmup,
            aug_mode=args.inference_mode if args.use_aug else 'none',

        )
        print('[INFO] train_loss:', train_loss)

        valid_acc = eval_overall(
            model, valid_loader, device, mode=args.inference_mode
        )
        test_acc = eval_overall(
            model, test_loader, device, mode=args.inference_mode
        )

        print(f'[INFO] valid: {valid_acc}, test: {test_acc}')
        log_info['train_loss'].append(train_loss)
        log_info['valid_metric'].append(valid_acc)
        log_info['test_metric'].append(test_acc)
        with open(log_dir, 'w') as Fout:
            json.dump(log_info, Fout, indent=4)

        if best_perf is None or valid_acc > best_perf:
            best_perf, best_epoch = valid_acc, ep
            torch.save(model.state_dict(), model_dir)

        if args.early_stop > 4 and ep > max(10, args.early_stop):
            val_his = log_info['valid_metric'][-args.early_stop:]
            if check_early_stop(val_his):
                break

        if ep >= args.warmup:
            scher.step()

    print('[Overall]')
    print('[bset_ep]', best_epoch)
    print('[best valid]', best_perf)
    print('[bset test]', log_info['test_metric'][best_epoch])
