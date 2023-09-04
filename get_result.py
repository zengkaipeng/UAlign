import json
import os
import argparse
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', required=True)
    args = parser.parse_args()

    bpref, btime, bargs = None, None, None
    for x in os.listdir(args.dir):
        if x.startswith('log-') and x.endswith('.json'):
            with open(os.path.join(args.dir, x)) as Fin:
                INFO = json.load(Fin)
            timestamp = x[4: -5]

        valid_all_covers = [x['all_cover'] for x in INFO['valid_metrics']]
        test_all_covers = [x['all_cover'] for x in INFO['test_metrics']]
        if len(valid_all_covers) == 0:
            continue

        best_idx = np.argmax(valid_all_covers)
        best_all_cover = test_all_covers[best_idx]

        if bpref is None or best_all_cover > bpref['all_cover']:
            bpref = INFO['test_metrics'][best_idx]
            btime, bargs = timestamp, INFO['args']

    print('[args]\n', bargs)
    print('[TIME]', btime)
    print('[pref]', bpref)
