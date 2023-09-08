import argparse
import os
import json

def filter_args(args, filter_ag):
    for k, v in filter_ag.items():
        if args.get(k, None) != v:
            return False
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Parser')
    parser.add_argument("--transformer", action='store_true')
    parser.add_argument('--log_dir', required=True)
    parser.add_argument('--gnn_type', choices=['gcn', 'gin', 'gat'])

    args = parser.parse_args()
    base_filter = {'gnn_type': args.gnn_type, 'transformer': args.transformer}
    model_name = ('Gtrans_' if args.transformer else '') + args.gnn_type

    finished = {}

    t_keys = ['dropout', 'mode', 'heads', 'update_gate', 'pos_enc']
    base_dir = os.path.join(args.log_dir, model_name)

    for x in os.listdir(base_dir):
        if x.startswith('log-') and x.endswith('.json'):
            with open(os.path.join(base_dir, x)) as Fin:
                INFO = json.load(Fin)

            if not filter_args(INFO['args'], base_filter):
                continue

            if INFO['args']['dim'] not in finished:
                finished[INFO['args']['dim']] = set()

            if args.transformer:
                o_key = t_keys
            elif args.gnn_type != 'gat':
                o_key = t_keys[:2]
            else:
                o_key = t_keys[:3]

            ix = tuple(INFO['args'][x] for x in o_key)
            finished[INFO['args']['dim']].add(ix)

    print(f'[MODEL] {model_name}')

    for k, v in finished.items():
        print(f'[DIM] {k}')
        for ix in v:
            print({t_keys[x]: ix[x] for x in range(len(ix))})
        print('\n')
