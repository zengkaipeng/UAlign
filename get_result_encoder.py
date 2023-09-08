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

            valid_all_covers = [x['all_cover'] for x in INFO['valid_metric']]
            test_all_covers = [x['all_cover'] for x in INFO['test_metric']]

            valid_all_hit = [x['all_fit'] for x in INFO['valid_metric']]
            test_all_hit = [x['all_fit'] for x in INFO['test_metric']]
            if len(valid_all_covers) == 0:
                continue

            best_idx = np.argmax(valid_all_covers)
            best_all_cover = test_all_covers[best_idx]

            if bpref is None or best_all_cover > bpref['all_cover']:
                bpref = INFO['test_metric'][best_idx]
                btime, bargs, bep = timestamp, INFO['args'], best_idx

            best_idx = np.argmax(valid_all_hit)
            best_all_hit = test_all_hit[best_idx]

            best_idx = np.argmax(valid_all_hit)
            if fpref is None or best_all_hit > fpref['all_fit']:
                fpref = INFO['test_metric'][best_idx]
                ftime, fargs, fep = timestamp, INFO['args'], best_idx

    print('[best_cover]')
    print('[args]\n', bargs)
    print('[TIME]', btime)
    print('[EPOCH]', bep)
    print('[pref]', bpref)

    print('[best_fit]')
    print('[args]\n', fargs)
    print('[TIME]', ftime)
    print('[EPOCH]', fep)
    print('[pref]', fpref)
