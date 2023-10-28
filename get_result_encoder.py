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
    fpref, ftime, fargs, fep = None, None, None, None

    for x in os.listdir(args.dir):
        if x.startswith('log-') and x.endswith('.json'):
            with open(os.path.join(args.dir, x)) as Fin:
                INFO = json.load(Fin)
            timestamp = x[4: -5]

            if not filter_args(INFO['args'], args_ft):
                continue

            node_val_fit = [x['by_node']['fit'] for x in INFO['valid_metric']]
            node_test_fit = [x['by_node']['fit'] for x in INFO['test_metric']]

            edge_val_fit = [x['by_edge']['fit'] for x in INFO['valid_metric']]
            edge_test_fit = [x['by_edge']['fit'] for x in INFO['test_metric']]

            if len(node_val_fit) == 0:
                continue

            best_idx = np.argmax(node_val_fit)
            best_node_fit = node_test_fit[best_idx]

            if bpref is None or best_node_fit > bpref['by_node']['fit']:
                bpref = INFO['test_metric'][best_idx]
                btime, bargs, bep = timestamp, INFO['args'], best_idx

            best_idx = np.argmax(edge_val_fit)
            best_edge_fit = edge_test_fit[best_idx]

            if fpref is None or best_edge_fit > fpref['by_edge']['fit']:
                fpref = INFO['test_metric'][best_idx]
                ftime, fargs, fep = timestamp, INFO['args'], best_idx

    print('[BEST BY NODE]')
    print('[args]\n', bargs)
    print('[TIME]', btime)
    print('[EPOCH]', bep)
    print('[pref]', bpref)

    print('[BEST BY EDGE]')
    print('[args]\n', fargs)
    print('[TIME]', ftime)
    print('[EPOCH]', fep)
    print('[pref]', fpref)