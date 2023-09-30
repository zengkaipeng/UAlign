import torch
import argparse
import json
import os
import time
import pickle


from tokenlizer import DEFAULT_SP, Tokenizer
from torch.utils.data import DataLoader
from sparse_backBone import GINBase, GATBase
from model import Graph2Seq, get_col_fc, PositionalEncoding, Acc_fn
from training import train_trans, eval_trans
from data_utils import load_ext_data, fix_seed
from torch.nn import TransformerDecoderLayer, TransformerDecoder
from torch.optim.lr_scheduler import ExponentialLR
from model import OnFlyDataset


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

    detail_log_folder = os.path.join(
        args.base_log,  args.backbone, '-'.join(log_dir)
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

    if not torch.cuda.is_available() or args.device < 0:
        device = torch.device('cpu')
    else:
        device = torch.device(f'cuda:{args.device}')

    fix_seed(args.seed)

    with open(args.token_path) as Fin:
        ALL_TOKEN = json.load(Fin)

    tokenizer = Tokenizer(ALL_TOKEN, DEFAULT_SP)

    train_rec, train_prod, train_rxn, train_target =\
        load_ext_data(args.data_path, 'train')
    val_rec, val_prod, val_rxn, val_target =\
        load_ext_data(args.data_path, 'val')
    test_rec, test_prod, test_rxn, test_target =\
        load_ext_data(args.data_path, 'test')

    print('[INFO] Data Loaded')

    train_set = OnFlyDataset(
        prod_sm=train_prod, reat_sm=train_rec, target=train_target,
        aug_prob=args.aug_prob, randomize=True,
    )
    valid_set = OnFlyDataset(
        prod_sm=val_prod, reat_sm=val_rec, target=val_target,
        aug_prob=0, randomize=False
    )
    test_set = OnFlyDataset(
        prod_sm=test_prod, reat_sm=test_rec, target=test_target,
        aug_prob=0, randomize=False
    )

    if args.backbone in ['GAT', 'MIX']:
        col_fn = get_col_fc(self_loop=True)
    else:
        col_fn = get_col_fc(self_loop=False)
    train_loader = DataLoader(
        train_set, collate_fn=col_fn,
        batch_size=args.bs, shuffle=True,
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
            num_layers=args.layer_encoder, dropout=args.dropout,
            embedding_dim=args.dim, edge_last=True, residual=True
        )
    else:
        GNN = GATBase(
            num_layers=args.layer_encoder, dropout=args.dropout,
            embedding_dim=args.dim, edge_last=True,
            residual=True, negative_slope=args.negative_slope,
            num_heads=args.heads, add_self_loop=False
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
    ).to(device)

    if args.checkpoint != '':
        assert args.token_ckpt != '', 'Missing Tokenizer Information'
        print(f'[INFO] Loading model weight in {args.checkpoint}')
        weight = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(weight)

    if args.token_ckpt != '':
        print(f'[INFO] Loading tokenizer from {args.token_ckpt}')
        with open(args.token_ckpt, 'rb') as Fin:
            tokenizer = pickle.load(Fin)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    lr_sh = ExponentialLR(
        optimizer, gamma=args.gamma, verbose=True
    )
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
    print('[INFO] padding index', tokenizer.token2idx['<PAD>'])

    log_info = {
        'args': args.__dict__, 'train_loss': [],
        'valid_metric': [], 'test_metric': []
    }

    with open(token_dir, 'wb') as Fout:
        pickle.dump(tokenizer, Fout)

    with open(log_dir, 'w') as Fout:
        json.dump(log_info, Fout, indent=4)

    for ep in range(args.epoch):
        print(f'[INFO] traing at epoch {ep + 1}')
        node_loss, edge_loss, tran_loss, tracc = train_trans(
            train_loader, model, optimizer, device, tokenizer,
            node_fn, edge_fn, tran_fn, acc_fn, verbose=True,
            warmup=(ep < args.warmup), accu=args.accu
        )
        log_info['train_loss'].append({
            'node': node_loss, 'edge': edge_loss,
            'trans': tran_loss, 'acc': tracc
        })

        valid_results, valid_acc = eval_trans(
            valid_loader, model, device, valid_fn,
            tokenizer, acc_fn, verbose=True
        )
        log_info['valid_metric'].append({
            'trans': valid_results, 'acc': valid_acc
        })

        test_results, test_acc = eval_trans(
            test_loader, model, device, valid_fn,
            tokenizer, acc_fn, verbose=True
        )

        log_info['test_metric'].append({
            'trans': test_results, 'acc': test_acc
        })

        print('[TRAIN]', log_info['train_loss'][-1])
        print('[VALID]', log_info['valid_metric'][-1])
        print('[TEST]', log_info['test_metric'][-1])

        if ep >= args.warmup and ep >= args.step_start:
            lr_sh.step()

        with open(log_dir, 'w') as Fout:
            json.dump(log_info, Fout, indent=4)
        if best_perf is None or valid_results < best_perf:
            best_perf, best_ep = valid_results, ep
            torch.save(model.state_dict(), model_dir)

        if best_acc is None or valid_acc > best_acc:
            best_acc, best_ep2 = valid_acc, ep
            torch.save(model.state_dict(), acc_dir)

        if args.early_stop > 3 and ep > max(10, args.early_stop):
            tx = [
                x['trans'] for x in
                log_info['valid_metric'][-args.early_stop:]
            ]
            ty = [
                x['acc'] for x in
                log_info['valid_metric'][-args.early_stop:]
            ]
            if all(x >= tx[0] for x in tx) and all(x <= ty[0] for x in ty):
                break

    print(f'[INFO] best loss epoch: {best_ep}')
    print(f'[INFO] best valid loss: {log_info["valid_metric"][best_ep]}')
    print(f'[INFO] best test loss: {log_info["test_metric"][best_ep]}')

    print(f'[INFO] best acc epoch: {best_ep2}')
    print(f'[INFO] best valid loss: {log_info["valid_metric"][best_ep2]}')
    print(f'[INFO] best test loss: {log_info["test_metric"][best_ep2]}')
