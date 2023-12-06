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

    bpref, btime, bargs, bep, bap = [None] * 5
    bloss, btime2, bargs2, bep2, bap2 = [None] * 5

    for x in os.listdir(args.dir):
        if x.startswith('log-') and x.endswith('.json'):
            with open(os.path.join(args.dir, x)) as Fin:
                INFO = json.load(Fin)
            timestamp = x[4: -5]

            if not filter_args(INFO['args'], args_ft):
                continue

            if len(INFO['valid_metric']) == 0:
                continue

            valid_losses = [x['all'] for x in INFO['valid_loss']]
            this_ep = np.argmin(valid_losses)

            if bloss is None or INFO['test_loss'][this_ep]['all'] < bloss:
                bloss = INFO['test_loss'][this_ep]['all']
                bep2, btime2, bargs2 = this_ep, timestamp, INFO['args']
                bap2 = {
                    'loss': INFO['test_loss'][this_ep],
                    'perf': INFO['test_metric'][this_ep]
                }

            valid_metric = [x['all'] for x in INFO['valid_metric']]
            this_ep = np.argmax(valid_metric)

            if bpref is None or INFO['test_metric'][this_ep]['all'] > bpref:
                bpref = INFO['test_metric'][this_ep]['all']
                bep, btime, bargs = this_ep, timestamp, INFO['args']
                bap = {
                    'loss': INFO['test_loss'][this_ep],
                    'perf': INFO['test_metric'][this_ep]
                }

    print('[BY ACC]')
    print('[args]\n', json.dumps(bargs, indent=4))
    print('[TIME]', btime)
    print('[EPOCH]', bep)
    print('[pref]', json.dumps(bap, indent=4))

    print('\n\n[BY LOSS]')
    print('[args]\n', json.dumps(bargs2, indent=4))
    print('[TIME]', btime2)
    print('[EPOCH]', bep2)
    print('[pref]', json.dumps(bap2, indent=4))
