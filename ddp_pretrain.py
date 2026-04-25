import torch
import argparse
import json


from torch.utils.data import DataLoader
from models import PretrainModel, load_model_arch
from utils.Dataset import TransDataset, col_fn_pretrain
from utils.training import ddp_pretrain, ddp_preeval
from utils.data_utils import (
    check_early_stop,
    create_log_model,
    dump_tokenizer,
    fix_seed,
    init_tokenizer,
    load_moles,
)
from torch.optim.lr_scheduler import ExponentialLR


import torch.distributed as torch_dist
import torch.multiprocessing as torch_mp
from torch.utils.data.distributed import DistributedSampler
def main_worker(worker_idx, args, tokenizer, log_dir, model_dir):

    print(f'[INFO] Process {worker_idx} start')
    torch_dist.init_process_group(
        backend='nccl', init_method=f'tcp://127.0.0.1:{args.port}',
        world_size=args.num_gpus, rank=worker_idx
    )

    device = torch.device(f'cuda:{worker_idx}')
    verbose = (worker_idx == 0)
    show_progress = verbose and (not args.scilence)

    train_moles, train_reac = load_moles(args.data_path, 'train', show_progress)
    test_moles, test_reac = load_moles(args.data_path, 'val', show_progress)

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

    model = PretrainModel.from_arch(
        token_size=tokenizer.get_token_size(),
        model_arch=args.model_arch,
        use_class=False,
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
            warmup=(ep < args.warmup), accu=args.accu,
            verbose=show_progress
        )

        test_results = ddp_preeval(
            loader=test_loader, model=model, tokenizer=tokenizer,
            pad_token='<PAD>', end_token='<END>', device=device,
            verbose=show_progress
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
        '--model_arch_path', required=True, type=str,
        help='the path of model architecture json'
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
        '--base_log', default='ddp_pretrain', type=str,
        help='the base dir of logging'
    )
    parser.add_argument(
        '--log_name', default='', type=str,
        help='the shared name for log/model/token outputs'
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
    parser.add_argument(
        '--scilence', action='store_true',
        help='disable tqdm progress bars while keeping epoch logs'
    )

    # training

    args = parser.parse_args()
    print(args)

    log_dir, model_dir, token_dir = create_log_model(
        args.base_log, args.log_name
    )
    tokenizer = init_tokenizer(
        token_path=args.token_path,
        checkpoint=args.checkpoint,
        token_ckpt=args.token_ckpt,
    )
    dump_tokenizer(tokenizer, token_dir)

    print(f'[INFO] padding index', tokenizer.token2idx['<PAD>'])
    fix_seed(args.seed)
    args.model_arch = load_model_arch(args.model_arch_path)

    torch_mp.spawn(
        main_worker, nprocs=args.num_gpus,
        args=(args, tokenizer, log_dir, model_dir)
    )


