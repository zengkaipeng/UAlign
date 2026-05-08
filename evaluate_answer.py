import argparse
import json
import os
import re

import numpy as np
from tqdm import tqdm

from utils.chemistry_parse import canonical_smiles, clear_map_number


CANON_CACHE = {}
REAL_ANSWER_CACHE = {}
RESULT_FILE_RE = re.compile(r'^\d+-\d+\.json$')
ARG_KEYS_TO_MATCH = (
    'model_arch_path',
    'data_path',
    'checkpoint',
    'token_ckpt',
    'use_class',
    'max_len',
    'beams',
    'batch_size',
    'aug_time',
    'disable_kv_cache',
    'rank_compute',
    'score_alpha',
)


def normalize_args(args):
    return {key: args.get(key) for key in ARG_KEYS_TO_MATCH}


def load_answers(path, single_file):
    if single_file:
        with open(path) as fin:
            info = json.load(fin)
        return info['answer'], info['args']

    answers = []
    saved_args = None
    for file_name in sorted(os.listdir(path)):
        if not RESULT_FILE_RE.match(file_name):
            continue
        with open(os.path.join(path, file_name)) as fin:
            info = json.load(fin)
        if 'answer' not in info or not isinstance(info['answer'], list):
            continue
        current_args = normalize_args(info.get('args', {}))
        if saved_args is None:
            saved_args = current_args
        elif current_args != saved_args:
            raise ValueError(
                f'inconsistent inference args detected in {file_name}'
            )
        answers.extend(info['answer'])
    if saved_args is None:
        raise FileNotFoundError(
            f'No shard result json found under {path}'
        )
    return answers, saved_args


def compute_topk_accuracy(answers, beam):
    topks = []
    for single in tqdm(answers):
        query = single['query']
        if query not in REAL_ANSWER_CACHE:
            reac, _ = query.split('>>')
            REAL_ANSWER_CACHE[query] = clear_map_number(reac)
        real_ans = REAL_ANSWER_CACHE[query]
        opt = np.zeros(beam)
        for idx, pred in enumerate(single['answer'][:beam]):
            if pred not in CANON_CACHE:
                CANON_CACHE[pred] = canonical_smiles(pred)
            if CANON_CACHE[pred] == real_ans:
                opt[idx:] = 1
                break
        topks.append(opt)
    topks = np.stack(topks, axis=0)
    return np.mean(topks, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--path', required=True, type=str,
        help='the path for file storing result'
    )
    parser.add_argument(
        '--top_k', nargs='+', type=int, default=[1, 3, 5, 10],
        help='top-k values to report, e.g. --top_k 1 3 5 10'
    )
    parser.add_argument(
        '--single_file', action='store_true',
        help='treat --path as a single json result file instead of a folder'
    )
    args = parser.parse_args()
    top_k = sorted(set(args.top_k))
    if not top_k or any(x <= 0 for x in top_k):
        raise ValueError('--top_k should be positive integers')
    beam = max(top_k)

    answers, saved_args = load_answers(args.path, args.single_file)
    topk_acc = compute_topk_accuracy(answers, beam)

    print(f'[args]\n{saved_args}')
    for i in top_k:
        print(f'[TOP {i}]', topk_acc[i - 1])


if __name__ == '__main__':
    main()
