import argparse
import json
import os

import numpy as np
from tqdm import tqdm

from utils.chemistry_parse import canonical_smiles, clear_map_number


def load_answers(path, single_file):
    if single_file:
        with open(path) as fin:
            info = json.load(fin)
        return info['answer'], info['args']

    answers = []
    saved_args = None
    for file_name in sorted(os.listdir(path)):
        if not file_name.endswith('.json'):
            continue
        with open(os.path.join(path, file_name)) as fin:
            info = json.load(fin)
        saved_args = info['args']
        answers.extend(info['answer'])
    if saved_args is None:
        raise FileNotFoundError(f'No json result file found under {path}')
    return answers, saved_args


def compute_topk_accuracy(answers, beam):
    topks = []
    for single in tqdm(answers):
        reac, _ = single['query'].split('>>')
        real_ans = clear_map_number(reac)
        opt = np.zeros(beam)
        for idx, pred in enumerate(single['answer'][:beam]):
            if canonical_smiles(pred) == real_ans:
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
        '--beam', type=int, default=10,
        help='the number of beams for searching'
    )
    parser.add_argument(
        '--single_file', action='store_true',
        help='treat --path as a single json result file instead of a folder'
    )
    args = parser.parse_args()

    answers, saved_args = load_answers(args.path, args.single_file)
    topk_acc = compute_topk_accuracy(answers, args.beam)

    print(f'[args]\n{saved_args}')
    for i in [1, 3, 5, 10]:
        if i <= args.beam:
            print(f'[TOP {i}]', topk_acc[i - 1])


if __name__ == '__main__':
    main()
