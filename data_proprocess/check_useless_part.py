"""
Canonicalize the product SMILES, and then use substructure matching to infer
the correspondence to the original atom-mapped order. This correspondence is then
used to renumber the reactant atoms.
"""

import rdkit
import os
import argparse
import pandas as pd
from rdkit import Chem


def check_use_less_part(rxn_smi):
    reac, prod = rxn_smi.split('>>')
    prod_mol = Chem.MolFromSmiles(prod)
    prod_amap = set(x.GetAtomMapNum() for x in prod_mol.GetAtoms())
    for y in reac.split('.'):
        y_mol = Chem.MolFromSmiles(y)
        y_amap = set(x.GetAtomMapNum() for x in y_mol.GetAtoms())
        if len(y_amap & prod_amap) == 0:
            return True
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--filename", required=True,
        help="File with reactions to canonicalize"
    )
    args = parser.parse_args()

    file_path = os.path.abspath(args.filename)
    file_name = os.path.basename(file_path)
    file_dir = os.path.dirname(file_path)
    new_file = f"canonicalized_{file_name}"
    df = pd.read_csv(args.filename)

    have_useless = 0

    new_dict = {'id': [], 'class': [], 'reactants>reagents>production': []}
    for idx in range(len(df)):
        element = df.loc[idx]
        rxn_smi = element['reactants>reagents>production']
        if check_use_less_part(rxn_smi):
            print('[FIND]', rxn_smi)
            have_useless += 1

    print(f'[prop] {have_useless} / {len(df)}={have_useless / len(df)}')


if __name__ == "__main__":
    main()
