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

    args = parser.parse_args()
    args_ft = eval(args.filter)

    bpref, btime, bargs, bep = None, None, None, None

    for x in os.listdir(args.dir):
        if x.startswith('log-') and x.endswith('.json'):
            with open(os.path.join(args.dir, x)) as Fin:
                INFO = json.load(Fin)
            timestamp = x[4: -5]

            if not filter_args(INFO['args'], args_ft):
                continue

            if len(INFO['valid_metric']) == 0:
                continue

            best_idx = np.argmax([x['edge'] for x in INFO['valid_metric']])
            best_node_fit = INFO['test_metric'][best_idx]['edge']

            if bpref is None or best_node_fit > bpref['edge']:
                bpref = INFO['test_metric'][best_idx]
                btime, bargs, bep = timestamp, INFO['args'], best_idx

    print('[args]\n', bargs)
    print('[TIME]', btime)
    print('[EPOCH]', bep)
    print('[pref]', bpref)
