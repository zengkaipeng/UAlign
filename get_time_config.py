import json
import os
import argparse
import numpy as np


def filter_args(args, filter_ag):
    for k, v in filter_ag.items():
        if args.get(k, None) != v:
            return False
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', required=True)
    parser.add_argument(
        '--filter', type=str, default='{}',
        help='a string as filter dict'
    )
    parser.add_argument(
        '--topk', type=int, default=10,
        help='the number of results to list'
    )

    args = parser.parse_args()
    args_ft = eval(args.filter)

    all_pfs = []

    for x in os.listdir(args.dir):
        if x.startswith('log-') and x.endswith('.json'):
            with open(os.path.join(args.dir, x)) as Fin:
                INFO = json.load(Fin)
            timestamp = x[4: -5]

            if not filter_args(INFO['args'], args_ft):
                continue

            if len(INFO['valid_metric']) == 0:
                continue

            best_idx = np.argmax([x['trans'] for x in INFO['valid_metric']])
            curr_perf = INFO['test_metric'][best_idx]

            all_pfs.append((INFO['args'], best_idx, timestamp, curr_perf))

    all_pfs.sort(key=lambda x: -x[-1]['trans'])
    for arg, ep, ts, pf in all_pfs[:args.topk]:
        print(f'[args]\n{arg}')
        print(f'[time] {ts}')
        print(f'[epoch] {ep}')
        print(f'[result] {pf}')
