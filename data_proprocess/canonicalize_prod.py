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
from tqdm import tqdm


def add_all_amap(rxn_smi):
    r, p = rxn_smi.split('>>')
    p_mol = Chem.MolFromSmiles(p)
    r_mol = Chem.MolFromSmiles(r)

    pmol_amaps = [x.GetAtomMapNum() for x in p_mol.GetAtoms()]
    pre_len = len(pmol_amaps)
    pmol_amaps = set(pmol_amaps)
    assert len(pmol_amaps) == pre_len and 0 not in pmol_amaps, \
        'Invalid atom mapping in the meta data'
    max_amap = max(pmol_amaps)

    for atom in r_mol.GetAtoms():
        amap_num = atom.GetAtomMapNum()
        if amap_num not in pmol_amaps:
            atom.SetAtomMapNum(max_amap + 1)
            max_amap += 1
    r_update = Chem.MolToSmiles(r_mol)
    return f"{r_update}>>{p}"


def remap_amap(rxn_smi):
    r, p = rxn_smi.split('>>')
    amap_remap = {}
    pmol = Chem.MolFromSmiles(p)
    rmol = Chem.MolFromSmiles(r)

    for atom in pmol.GetAtoms():
        xnum = atom.GetAtomMapNum()
        if xnum not in amap_remap:
            amap_remap[xnum] = len(amap_remap) + 1

    for atom in rmol.GetAtoms():
        xnum = atom.GetAtomMapNum()
        if xnum not in amap_remap:
            amap_remap[xnum] = len(amap_remap) + 1

    for atom in pmol.GetAtoms():
        xnum = atom.GetAtomMapNum()
        atom.SetAtomMapNum(amap_remap[xnum])

    for atom in rmol.GetAtoms():
        xnum = atom.GetAtomMapNum()
        atom.SetAtomMapNum(amap_remap[xnum])

    r_update = Chem.MolToSmiles(rmol)
    p_update = Chem.MolToSmiles(pmol)
    return f"{r_update}>>{p_update}"


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
    print(f"Processing file of size: {len(df)}")

    new_dict = {'id': [], 'class': [], 'reactants>reagents>production': []}
    for idx in tqdm(range(len(df))):
        element = df.loc[idx]
        uspto_id, class_id = element['id'], element['class']
        rxn_smi = element['reactants>reagents>production']

        rxn_new = add_all_amap(rxn_smi)
        rxn_new = remap_amap(rxn_new)
        new_dict['id'].append(uspto_id)
        new_dict['class'].append(class_id)
        new_dict['reactants>reagents>production'].append(rxn_new)

    new_df = pd.DataFrame.from_dict(new_dict)
    new_df.to_csv(f"{file_dir}/{new_file}", index=False)


if __name__ == "__main__":
    main()
