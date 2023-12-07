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
from torch.optim.lr_scheduler import MultiStepLR


def create_log_model(args):
    timestamp = time.time()
    log_dir = [
        f'dim_{args.dim}', f'seed_{args.seed}', f'dropout_{args.dropout}',
        f'bs_{args.bs}', f'lr_{args.lr}', f'heads_{args.heads}',
        f'encoder_{args.layer_encoder}', f'decoder_{args.layer_decoder}',
        f'label_smooth_{args.label_smooth}'
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
    return detail_log_dir, detail_model_dir, token_dir


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
        '--backbone', type=str, choices=['GAT', 'GIN'],
        help='type of gnn backbone', required=True
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
        '--label_smooth', default=0.0, type=float,
        help='the label smoothing for transformer'
    )

    args = parser.parse_args()
    print(args)
    log_dir, model_dir, token_dir = create_log_model(args)

    if not torch.cuda.is_available() or args.device < 0:
        device = torch.device('cpu')
    else:
        device = torch.device(f'cuda:{args.device}')

    fix_seed(args.seed)

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

    print('[INFO] Data Loaded')

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
        train_set, collate_fn=fc_collect_fn,
        batch_size=args.bs, shuffle=True
    )
    valid_loader = DataLoader(
        valid_set, collate_fn=fc_collect_fn,
        batch_size=args.bs, shuffle=False
    )
    test_loader = DataLoader(
        test_set, collate_fn=fc_collect_fn,
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
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    lr_sh = MultiStepLR(
        optimizer, milestones=[150, 300, 600],
        gamma=0.5, verbose=True
    )
    best_perf, best_ep = None, None

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
        node_loss, edge_loss, tran_loss = train_trans(
            train_loader, model, optimizer, device, tokenizer,
            node_fn, edge_fn, tran_fn, verbose=True, warmup=(ep == 0)
        )
        log_info['train_loss'].append({
            'node': node_loss, 'edge': edge_loss, 'trans': tran_loss
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

        lr_sh.step()

        with open(log_dir, 'w') as Fout:
            json.dump(log_info, Fout, indent=4)
        if best_perf is None or valid_results < best_perf:
            best_perf, best_ep = valid_results, ep
            torch.save(model.state_dict(), model_dir)

        if args.early_stop > 5 and ep > max(20, args.early_stop):
            tx = [
                x['trans'] for x in
                log_info['valid_metric'][-args.early_stop:]
            ]
            if all(x >= tx[0] for x in tx):
                break

    print(f'[INFO] best epoch: {best_ep}')
    print(f'[INFO] best valid loss: {log_info["valid_metric"][best_ep]}')
    print(f'[INFO] best test loss: {log_info["test_metric"][best_ep]}')
