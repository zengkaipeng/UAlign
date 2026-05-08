"""
Canonicalize USPTO-MIT text splits.

USPTO-MIT uses the shared chemistry canonicalization helpers, but keeps a
dataset-specific input flow and removes reactant fragments unrelated to the
product atom maps.
"""

import argparse
import os

import pandas as pd
from rdkit import Chem

from canonicalize_data_csv import add_all_amap, check_valid, remap_amap


def clear_useless_part(reaction):
    reac, prod = reaction.split('>>')
    pmol = Chem.MolFromSmiles(prod)
    p_amps = set(x.GetAtomMapNum() for x in pmol.GetAtoms())
    x_reac = []
    for x in reac.split('.'):
        rmol = Chem.MolFromSmiles(x)
        r_amps = set(x.GetAtomMapNum() for x in rmol.GetAtoms())
        if len(p_amps & r_amps) == 0:
            continue
        x_reac.append(x)
    return f"{'.'.join(x_reac)}>>{prod}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir", required=True,
        help="directory containing train.txt, valid.txt, and test.txt"
    )
    parser.add_argument(
        '--output_dir', required=True,
        help='the path of processed result'
    )
    args = parser.parse_args()

    id_marker = 0
    for part in ['train', 'valid', 'test']:
        new_dict = {'id': [], 'class': [], 'reactants>reagents>production': []}
        filename = os.path.join(args.dir, f'{part}.txt')
        out_name = f'canonicalized_raw_{part if part != "valid" else "val"}.csv'
        outfile = os.path.join(args.output_dir, out_name)
        raw_len = 0
        with open(filename) as Fin:
            for lin in Fin:
                if len(lin) <= 1:
                    continue
                try:
                    reaction = lin.split()[0].strip()
                    reaction.split('>>')
                except Exception as e:
                    print('[Error Line]', lin)
                    print(e)
                    continue
                raw_len += 1
                is_valid, message = check_valid(
                    reaction, reject_multi_product=True
                )
                if not is_valid:
                    print('[reaction]', reaction)
                    print('[message]', message)
                    continue

                rxn_new = add_all_amap(reaction)
                rxn_new = clear_useless_part(rxn_new)
                rxn_new = remap_amap(rxn_new)
                id_marker += 1
                new_dict['id'].append(f'mit_{id_marker}')
                new_dict['class'].append(-1)
                new_dict['reactants>reagents>production'].append(rxn_new)

        new_df = pd.DataFrame.from_dict(new_dict)
        new_df.to_csv(outfile, index=False)
        print('[INFO] processed part:', part)
        print('[INFO] file size before process:', raw_len)
        print('[INFO] file size after process:', len(new_df))


if __name__ == "__main__":
    main()
