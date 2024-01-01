from utils.chemistry_parse import canonical_smiles, clear_map_number
import json
import numpy as np
import argparse
from tqdm import tqdm


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

    args = parser.parse_args()

    with open(args.path) as Fin:
        answers = json.load(Fin)

    print(f'[args]\n{answers["args"]}')

    topks = []
    for single in tqdm(answers['answer']):
        reac, prod = single['query'].split('>>')
        real_ans = clear_map_number(reac)
        opt = np.zeros(args.beam)
        for idx, x in enumerate(single['answer']):
            x = canonical_smiles(x)
            if x == real_ans:
                opt[idx:] = 1
                break
        topks.append(opt)
    topks = np.stack(topks, axis=0)
    topk_acc = np.mean(topks, axis=0)

    for i in [1, 3, 5, 10]:
        print(f'[TOP {i}]', topk_acc[i - 1])
