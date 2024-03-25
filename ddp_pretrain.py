import pickle
import torch
import argparse
import json
import os
import time


from torch.utils.data import DataLoader
from sparse_backBone import GATBase
from Dataset import TransDataset, col_fn_pretrain
from model import PositionalEncoding, PretrainModel
from ddp_training import ddp_pretrain, ddp_preeval
from data_utils import fix_seed, check_early_stop
from tokenlizer import DEFAULT_SP, Tokenizer
from torch.optim.lr_scheduler import ExponentialLR
from utils.chemistry_parse import clear_map_number
import pandas
from tqdm import tqdm

from torch.nn import TransformerDecoderLayer, TransformerDecoder


import torch.distributed as torch_dist
import torch.multiprocessing as torch_mp
from torch.utils.data.distributed import DistributedSampler


def create_log_model(args):
    timestamp = time.time()
    if not os.path.exists(args.base_log):
        os.makedirs(args.base_log)
    detail_log_dir = os.path.join(args.base_log, f'log-{timestamp}.json')
    detail_model_dir = os.path.join(args.base_log, f'mod-{timestamp}.pth')
    token_path = os.path.join(args.base_log, f'token-{timestamp}.pkl')
    return detail_log_dir, detail_model_dir, token_path


def load_moles(data_dir, part, verbose):
    df_train = pandas.read_csv(
        os.path.join(data_dir, f'canonicalized_raw_{part}.csv')
    )
    moles, reacts = set(), set()
    iterx = df_train['reactants>reagents>production']
    if verbose:
        iterx = tqdm(iterx)
    for idx, resu in enumerate(iterx):
        rea, prd = resu.strip().split('>>')
        rea = clear_map_number(rea)
        prd = clear_map_number(prd)
        moles.update(rea.split('.'))
        moles.add(prd)
        if '.' in rea:
            reacts.add(rea)
    return list(moles), list(reacts)


def main_worker(worker_idx, args, tokenizer, log_dir, model_dir):

    print(f'[INFO] Process {worker_idx} start')
    torch_dist.init_process_group(
        backend='nccl', init_method=f'tcp://127.0.0.1:{args.port}',
        world_size=args.num_gpus, rank=worker_idx
    )

    device = torch.device(f'cuda:{worker_idx}')
    verbose = (worker_idx == 0)

    train_moles, train_reac = load_moles(args.data_path, 'train', verbose)
    test_moles, test_reac = load_moles(args.data_path, 'val', verbose)

    print(f'[INFO] worker {worker_idx} data loaded')

    train_set = TransDataset(train_moles, train_reac, mode='train')
    test_set = TransDataset(test_moles, test_reac, mode='eval')

    train_sampler = DistributedSampler(train_set, shuffle=True)
    test_sampler = DistributedSampler(test_set, shuffle=False)

    train_loader = DataLoader(
        train_set, collate_fn=col_fn_pretrain, batch_size=args.bs,
        shuffle=False, pin_memory=True, sampler=train_sampler,
        num_workers=args.num_workers
    )
    test_loader = DataLoader(
        test_set, collate_fn=col_fn_pretrain,  batch_size=args.bs,
        shuffle=False, pin_memory=True, sampler=test_sampler,
        num_workers=args.num_workers
    )

    GNN = GATBase(
        num_layers=args.n_layer, dropout=args.dropout,
        embedding_dim=args.dim, num_heads=args.heads,
        negative_slope=args.negative_slope, n_class=None
    )

    decode_layer = TransformerDecoderLayer(
        d_model=args.dim, nhead=args.heads, batch_first=True,
        dim_feedforward=args.dim * 2, dropout=args.dropout
    )
    Decoder = TransformerDecoder(decode_layer, args.n_layer)
    Pos_env = PositionalEncoding(args.dim, args.dropout, maxlen=2000)

    model = PretrainModel(
        token_size=tokenizer.get_token_size(), encoder=GNN,
        decoder=Decoder, d_model=args.dim, pos_enc=Pos_env
    ).to(device)

    if args.checkpoint != '':
        weight = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(weight)

    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[worker_idx], output_device=worker_idx,
        find_unused_parameters=True
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    lr_scher = ExponentialLR(optimizer, args.lrgamma, verbose=verbose)
    best_cov, best_ep = None, None

    log_info = {
        'args': args.__dict__, 'train_loss': [],
        'test_metric': []
    }

    if verbose:
        with open(log_dir, 'w') as Fout:
            json.dump(log_info, Fout, indent=4)

    for ep in range(args.epoch):
        if verbose:
            print(f'[INFO] traing at epoch {ep + 1}')

        train_sampler.set_epoch(ep)
        loss = ddp_pretrain(
            loader=train_loader, model=model, optimizer=optimizer,
            tokenizer=tokenizer, device=device, pad_token='<PAD>',
            warmup=(ep < args.warmup), accu=args.accu, verbose=verbose
        )

        test_results = ddp_preeval(
            loader=test_loader, model=model, tokenizer=tokenizer,
            pad_token='<PAD>', end_token='<END>', device=device,
            verbose=verbose
        )
        torch_dist.barrier()
        loss.all_reduct(device)
        test_results.all_reduct(device)

        log_info['train_loss'].append(loss.get_all_value_dict())
        log_info['test_metric'].append(test_results.get_all_value_dict())

        if verbose:
            print('[TRAIN]', log_info['train_loss'][-1])
            print('[TEST]', log_info['test_metric'][-1])
            test_tacc = log_info['test_metric'][-1]['trans_acc']

            with open(log_dir, 'w') as Fout:
                json.dump(log_info, Fout, indent=4)
            if best_cov is None or test_tacc > best_cov:
                best_cov, best_ep = test_tacc, ep
                torch.save(model.module.state_dict(), model_dir)

        if ep >= args.warmup:
            lr_scher.step()

        if args.early_stop >= 5 and ep > max(10, args.early_stop):
            val_his = log_info['test_metric'][-args.early_stop:]
            val_his = [x['trans_acc'] for x in val_his]
            if check_early_stop(val_his):
                print(f'[INFO {worker_idx}] early_stop_break')
                break

    if not verbose:
        return

    print('[BEST EP]', best_ep)
    print('[BEST TEST]', log_info['test_metric'][best_ep])


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DDP first stage')
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
        '--lr', default='1e-3', type=float,
        help='the learning rate for training'
    )
    parser.add_argument(
        '--dropout', type=float, default=0.1,
        help='the dropout rate, useful for all backbone'
    )

    parser.add_argument(
        '--base_log', default='ddp_pretrain', type=str,
        help='the base dir of logging'
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
        '--token_path', type=str, default='',
        help='the path of json containing tokens'
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
    parser.add_argument(
        '--accu', type=int, default=1,
        help='the gradient accumulation step'
    )
    parser.add_argument(
        '--num_workers', default=0, type=int,
        help='the number of workers for dataloader per worker'
    )
    parser.add_argument(
        '--num_gpus', type=int, default=1,
        help='the number of gpus to train and eval'
    )
    parser.add_argument(
        '--port', type=int, default=12345,
        help='the port for ddp nccl communication'
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

    with open(token_dir, 'wb') as Fout:
        pickle.dump(tokenizer, Fout)

    print(f'[INFO] padding index', tokenizer.token2idx['<PAD>'])
    fix_seed(args.seed)

    torch_mp.spawn(
        main_worker, nprocs=args.num_gpus,
        args=(args, tokenizer, log_dir, model_dir)
    )
