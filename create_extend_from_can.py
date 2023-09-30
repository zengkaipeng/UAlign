from utils.chemistry_parse import clear_map_number
import argparse
import pandas
from tqdm import tqdm
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('--data_path', required=True)
    args = parser.parse_args()

    HEADER = 'id,class,reactants>reagents>production,clean_reactant\n'
    for part in ['train', 'val', 'test']:
        org_data = pandas.read_csv(os.path.join(
            args.data_path, f'canonicalized_raw_{part}.csv'
        ))
        out_name = os.path.join(args.data_path, f'extend_{part}.csv')

        with open(out_name, 'w') as Fout:
            Fout.write(HEADER)
            for idx in tqdm(range(len(org_data))):
                ID = org_data.loc[idx, 'id']
                CLS = org_data.loc[idx, 'class']
                rxn = org_data.loc[idx, 'reactants>reagents>production']
                react, prod = rxn.split('>>')
                clean_react = clear_map_number(react)
                Fout.write('{},{},{},{}\n'.format(ID, CLS, rxn, clean_react))
