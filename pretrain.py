import pickle
import torch
import argparse
import json
import os
import time

from torch.utils.data import DataLoader
from sparse_backBone import GINBase, GATBase
from Mix_backbone import MixFormer
from model import col_fn_pretrain, TransDataset
from training import pretrain, preeval
from data_utils import fix_seed, check_early_stop
from tokenlizer import DEFAULT_SP, Tokenizer
from torch.optim.lr_scheduler import ExponentialLR
from utils.chemistry_parse import clear_map_number
import pandas


def create_log_model(args):
    timestamp = time.time()
    detail_log_folder = os.path.join(
        args.base_log, ('Gtrans_' if args.transformer else '') + args.gnn_type
    )
    if not os.path.exists(detail_log_folder):
        os.makedirs(detail_log_folder)
    detail_log_dir = os.path.join(detail_log_folder, f'log-{timestamp}.json')
    detail_model_dir = os.path.join(detail_log_folder, f'mod-{timestamp}.pth')
    token_path = os.path.join(detail_log_folder, f'token-{timestamp}.pkl')
    return detail_log_dir, detail_model_dir, token_path


def load_moles(data_dir, part):
    df_train = pandas.read_csv(
        os.path.join(data_dir, f'canonicalized_raw_{part}.csv')
    )
    moles = []
    for idx, resu in enumerate(df_train['reactants>reagents>production']):
        rea, prd = resu.strip().split('>>')
        rea = clear_map_number(rea)
        prd = clear_map_number(prd)
        moles.extend(rea.split('.'))
        moles.append(prd)
    return moles


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Graph Edit Exp, Sparse Model')
    # public setting
    parser.add_argument(
        '--dim', default=256, type=int,
        help='the hidden dim of model'
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
        '--gnn_type', type=str, choices=['GAT', 'GIN', 'MIX'],
        help='type of gnn backbone', required=True
    )
    parser.add_argument(
        '--dropout', type=float, default=0.1,
        help='the dropout rate, useful for all backbone'
    )

    parser.add_argument(
        '--base_log', default='log_pretrain', type=str,
        help='the base dir of logging'
    )

    # GAT & Gtrans setting
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
        '--token_path', type=str, default='',
        help='the path of json containing tokens'
    )
    parser.add_argument(
        '--aug_prob', type=float, default=0.0,
        help='the augument prob for training'
    )
    parser.add_argument(
        '--checkpoint', type=str, default='',
        help='the checkpoint for pretrained model'
    )
    parser.add_argument(
        '--token_ckpt', type=str, default='',
        help='the path of token checkpoint, required while' +
        ' checkpoint is specified'
    )
    parser.add_argument(
        '--lrgamma', type=float, default=1,
        help='the gamma for lr_scheduler weight decay'
    )
    parser.add_argument(
        '--warmup', type=int, default=4,
        help='the epochs of warmup epochs'
    )

    # training

    args = parser.parse_args()
    print(args)

    log_dir, model_dir, token_dir = create_log_model(args)

    if args.checkpoint != '':
        assert args.token_ckpt != '', \
            'require token_ckpt when checkpoint is given'
        with open(args.token_ckpt, 'rb') as Fin:
            tokenizer = pickle.load(Fin)
    else:
        assert args.token_path != '', 'file containing all tokens are required'
        SP_TOKEN = DEFAULT_SP | set([f"<RXN>_{i}" for i in range(11)])

        with open(args.token_path) as Fin:
            tokenizer = Tokenizer(json.load(Fin), SP_TOKEN)

    if not torch.cuda.is_available() or args.device < 0:
        device = torch.device('cpu')
    else:
        device = torch.device(f'cuda:{args.device}')

    fix_seed(args.seed)

    train_moles = load_moles(args.data_path, 'train')
    test_moles = load_moles(args.data_path, 'val')

    train_set = PretrainDataset(train_moles, args.aug_prob)
    test_set = PretrainDataset(test_moles, random_prob=0)

    train_loader = DataLoader(
        train_set, collate_fn=pretrain_col_fn,
        batch_size=args.bs, shuffle=True
    )
    test_loader = DataLoader(
        test_set, collate_fn=pretrain_col_fn,
        batch_size=args.bs, shuffle=False
    )

    if args.backbone == 'GIN':
        GNN = GINBase(
            num_layers=args.layer_encoder, dropout=args.dropout,
            embedding_dim=args.dim,
        )
    elif args.backbone == 'GAT':
        GNN = GATBase(
            num_layers=args.layer_encoder, dropout=args.dropout,
            embedding_dim=args.dim, negative_slope=args.negative_slope,
            num_heads=args.heads,
        )
    else:
        GNN = MixFormer(
            emb_dim=args.dim, num_layer=args.layer_encoder,
            heads=args.heads, dropout=args.dropout,
            negative_slope=args.negative_slope,
        )

    trans_head = args.heads if args.backbone != 'MIX' else args.heads * 2

    decode_layer = TransformerDecoderLayer(
        d_model=args.dim, nhead=trans_head, batch_first=True,
        dim_feedforward=args.dim * 2, dropout=args.dropout
    )
    Decoder = TransformerDecoder(decode_layer, args.layer_decoder)
    Pos_env = PositionalEncoding(args.dim, args.dropout, maxlen=2000)

    model = Graph2Seq(
        token_size=tokenizer.get_token_size(), encoder=GNN,
        decoder=Decoder, d_model=args.dim, pos_enc=Pos_env
    ).to(device)

    if args.checkpoint != '':
        weight = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(weight)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    lr_scher = ExponentialLR(optimizer, args.lrgamma, verbose=True)
    best_cov, best_ep = None, None

    log_info = {
        'args': args.__dict__, 'train_loss': [],
        'test_metric': []
    }

    with open(log_dir, 'w') as Fout:
        json.dump(log_info, Fout, indent=4)
    with open(token_dir, 'wb') as Fout:
        pickle.dump(tokenizer, Fout)

    for ep in range(args.epoch):
        print(f'[INFO] traing at epoch {ep + 1}')
        loss = pretrain(
            loader=train_loader, model=model, optimizer=optimizer,
            tokenizer=tokenizer, device=device, pad_token='<PAD>',
            warmup=(ep < args.warmup),
        )
        log_info['train_loss'].append(loss)

        print('[TRAIN]', log_info['train_loss'][-1])

        test_results = preeval(
            loader=test_loader, model=model, tokenizer=tokenizer,
            pad_token='<PAD>', end_token='<END>', device=device
        )
        log_info['test_metric'].append(test_results)

        print('[TEST]', log_info['test_metric'][-1])

        if ep >= args.warmup:
            lr_scher.step()

        with open(log_dir, 'w') as Fout:
            json.dump(log_info, Fout, indent=4)

        if best_cov is None or test_results > best_cov:
            best_cov, best_ep = test_results, ep
            torch.save(model.state_dict(), model_dir)

        if args.early_stop > 5 and ep > max(20, args.early_stop):
            val_his = log_info['test_metric'][-args.early_stop:]
            if check_early_stop(val_his):
                break

    print('[BEST EP]', best_ep)
    print('[BEST TEST]', log_info['test_metric'][best_ep])
