import os
import argparse
import pandas

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Parser for data stat')
    parser.add_argument(
        '--dir', required=True,
        help='the path containing the dataset'
    )
    args = parser.parse_args()

    file_names = [f'canonicalized_raw_{p}.csv' for p in ['train', 'val', 'test']]

    cnts = {
        'train': [0, 0, 0], 'val': [0, 0, 0],
        'test': [0, 0, 0], 'total': [0, 0, 0]
    }

    for f in file_names:
        df = pandas.read_csv(os.path.join(args.dir, f))
        part = f.split('.')[0].split('_')[-1]
        for rxn in df['reactants>reagents>production']:
            reac, prod = rxn.strip().split('>>')
            reac = reac.split('.')
            if len(reac) == 1:
                cnts[part][0] += 1
                cnts['total'][0] += 1
            elif len(reac) == 2:
                cnts[part][1] += 1
                cnts['total'][1] += 1
            else:
                cnts[part][2] += 1
                cnts['total'][2] += 1

    table = [['', 'single', 'double', 'multiple']]
    for part in ['train', 'val', 'test', 'total']:
        line, sm = [part], sum(cnts[part])
        line.extend('{}/{}={:.3f}'.format(x, sm, x / sm) for x in cnts[part])
        table.append(line)

    split_line, line_format = [], []
    for i in range(4):
        max_len = max(len(table[x][i]) for x in range(len(table)))
        split_line.append('-' * (max_len + 2))
        line_format.append('{:^%d}' % (max_len + 2))

    split_line = '+{}+'.format('+'.join(split_line))
    line_format = '|{}|'.format('|'.join(line_format))
    print(split_line)
    for line in table:
        print(line_format.format(*line))
        print(split_line)

