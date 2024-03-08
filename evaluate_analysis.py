from utils.chemistry_parse import canonical_smiles, clear_map_number
import json
import numpy as np
import argparse
from tqdm import tqdm
import os


if __name__ == '__main__':
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
        '--isdir', action='store_true',
        help='whether the provided results is a folder'
    )
    args = parser.parse_args()

    if args.isdir:
        answers, targs = [], None
        for x in os.listdir(args.path):
            if x.endswith('.json'):
                with open(os.path.join(args.path, x)) as Fin:
                    INFO = json.load(Fin)
                targs = INFO['args']
                answers.extend(INFO['answer'])
    else:
        with open(args.path) as Fin:
            answers = json.load(Fin)
        targs = answers['args']
        answers = answers['answer']

    topks = {'total': [], 'single': [], 'double': [], 'multiple': []}
    for single in tqdm(answers):
        reac, prod = single['query'].split('>>')
        num_reac = len(reac.split('.'))
        real_ans = clear_map_number(reac)
        opt = np.zeros(args.beam)
        for idx, x in enumerate(single['answer']):
            x = canonical_smiles(x)
            if x == real_ans:
                opt[idx:] = 1
                break
        topks['total'].append(opt)
        if num_reac == 1:
            topks['single'].append(opt)
        elif num_reac == 2:
            topks['double'].append(opt)
        else:
            topks['multiple'].append(opt)

    xnum = {k: len(v) for k, v in topks.items()}
    topks = {k: np.stack(v, axis=0) for k, v in topks.items()}
    res = {k: np.mean(v, axis=0) for k, v in topks.items()}
    crr_sum = {k: np.sum(v, axis=0, dtype=np.int64) for k, v in topks.items()}

    table = [['', 'top-1', 'top-3', 'top-5', 'top-10']]
    for x in ['single', 'double', 'multiple', 'total']:
        line = [x]
        for i in [1, 3, 5, 10]:
            line.append('{} / {} = {:.2f}%'.format(
                crr_sum[x][i - 1], xnum[x], res[x][i - 1] * 100
            ))
        table.append(line)

    split_line, line_format = [], []
    for i in range(5):
        max_len = max(len(table[x][i]) for x in range(len(table)))
        split_line.append('-' * (max_len + 2))
        line_format.append('{:^%d}' % (max_len + 2))

    split_line = '+{}+'.format('+'.join(split_line))
    line_format = '|{}|'.format('|'.join(line_format))
    print(split_line)
    for line in table:
        print(line_format.format(*line))
        print(split_line)
