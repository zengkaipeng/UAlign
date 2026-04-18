import argparse
import json
import os
import pickle

import pandas
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import PretrainModel, load_model_arch
from utils.Dataset import InferenceDataset, col_fn_inference
from utils.data_utils import fix_seed
from utils.inference_tools import beam_search_batch
from utils.rerank import rerank_predictions


def build_model(args, tokenizer, device):
    model = PretrainModel.from_arch(
        token_size=tokenizer.get_token_size(),
        model_arch=args.model_arch,
        use_class=args.use_class,
    ).to(device)

    weight = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(weight, strict=False)
    model.eval()
    return model


def build_dataloader(args):
    meta_df = pandas.read_csv(args.data_path)
    end_pos = len(meta_df) if args.len <= 0 else min(len(meta_df), args.start + args.len)
    part_df = meta_df.iloc[args.start:end_pos]
    rxn_cls = part_df['class'].tolist() if args.use_class else None
    dataset = InferenceDataset(
        queries=part_df['reactants>reagents>production'].tolist(),
        indexes=part_df.index.tolist(),
        rxn_cls=rxn_cls,
        aug_time=args.aug_time,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=col_fn_inference,
    )
    return loader, end_pos


def dump_answers(path, args, answers):
    with open(path, 'w') as fout:
        json.dump({
            'args': args.__dict__,
            'answer': answers,
        }, fout, indent=4)


def main():
    parser = argparse.ArgumentParser('Batch inference')
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
        '--device', default=-1, type=int,
        help='the device for running exps'
    )
    parser.add_argument(
        '--checkpoint', type=str, required=True,
        help='the path of checkpoint to restart the exp'
    )
    parser.add_argument(
        '--token_ckpt', type=str, required=True,
        help='the path of tokenizer, when ckpt is loaded, necessary'
    )
    parser.add_argument(
        '--use_class', action='store_true',
        help='use the class for model or not'
    )
    parser.add_argument(
        '--max_len', default=300, type=int,
        help='the max num of tokens in result'
    )
    parser.add_argument(
        '--beams', default=10, type=int,
        help='the number of beams'
    )
    parser.add_argument(
        '--output_folder', default='results', type=str,
        help='the path containing results'
    )
    parser.add_argument(
        '--save_every', type=int, default=1000,
        help='the step to save result into file'
    )
    parser.add_argument(
        '--start', type=int, default=0,
        help='the start index for inference'
    )
    parser.add_argument(
        '--len', type=int, default=-1,
        help='the number of samples to run; negative means all remaining'
    )
    parser.add_argument(
        '--batch_size', type=int, default=32,
        help='the batch size for batched decoding'
    )
    parser.add_argument(
        '--aug_time', type=int, default=1,
        help='the number of product SMILES augmentations per sample'
    )
    parser.add_argument(
        '--num_workers', type=int, default=0,
        help='the number of workers for the inference dataloader'
    )
    parser.add_argument(
        '--disable_kv_cache', action='store_true',
        help='disable KV cache and recompute the decoder state each step'
    )
    parser.add_argument(
        '--rank_compute', type=str, default='log_logits',
        choices=['log_logits', 'ensemble'],
        help=(
            'how to combine augmented predictions: '
            'log_logits sums probabilities, ensemble uses reciprocal-rank '
            'voting after per-augmentation canonical merging'
        )
    )
    parser.add_argument(
        '--score_alpha', type=float, default=0.1,
        help='alpha used by ensemble reciprocal-rank scoring'
    )
    args = parser.parse_args()
    print(args)

    if not torch.cuda.is_available() or args.device < 0:
        device = torch.device('cpu')
    else:
        device = torch.device(f'cuda:{args.device}')

    fix_seed(args.seed)
    with open(args.token_ckpt, 'rb') as fin:
        tokenizer = pickle.load(fin)
    args.model_arch = load_model_arch(args.model_arch_path)

    model = build_model(args, tokenizer, device)
    loader, end_pos = build_dataloader(args)

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
    out_file = os.path.join(args.output_folder, f'{args.start}-{end_pos}.json')

    answers = []
    processed = 0
    for graphs, begin_tokens, queries, rxn_classes, indexes, aug_sizes in tqdm(loader):
        graphs = graphs.to(device)
        pred_batch, prob_batch = beam_search_batch(
            model=model,
            tokenizer=tokenizer,
            graphs=graphs,
            device=device,
            begin_tokens=begin_tokens,
            max_len=args.max_len,
            size=args.beams,
            pen_para=0,
            validate=False,
            use_kv_cache=not args.disable_kv_cache,
        )

        offset = 0
        for query, rxn_class, data_idx, aug_size in zip(
            queries, rxn_classes, indexes, aug_sizes
        ):
            preds, probs = rerank_predictions(
                pred_batch[offset: offset + aug_size],
                prob_batch[offset: offset + aug_size],
                rank_compute=args.rank_compute,
                keep_invalid=False,
                score_alpha=args.score_alpha,
            )
            answers.append({
                'query': query,
                'idx': int(data_idx),
                'rxn_class': rxn_class,
                'answer': preds,
                'prob': probs,
            })
            processed += 1
            offset += aug_size

        if args.save_every > 0 and processed % args.save_every == 0:
            dump_answers(out_file, args, answers)

    dump_answers(out_file, args, answers)


if __name__ == '__main__':
    main()
