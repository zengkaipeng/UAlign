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


def get_cano_ams(x):
    mol = Chem.MolFromSmiles(x)
    idx2am = {p.GetIdx(): p.GetAtomMapNum() for p in mol.GetAtoms()}
    for atom in mol.GetAtoms():
        if atom.HasProp('molAtomMapNumber'):
            atom.ClearProp('molAtomMapNumber')
    ranks = list(Chem.CanonicalRankAtoms(mol))
    y = list(range(len(ranks)))
    y.sort(key=lambda t: ranks[t])
    return [idx2am[t] for t in y]


def remap_using_cano(x):
    cano_ranks = get_cano_ams(x)
    remap = {v: idx + 1 for idx, v in enumerate(cano_ranks)}
    return remap


def remap_amap(rxn_smi):
    r, p = rxn_smi.split('>>')
    pmol = Chem.MolFromSmiles(p)
    amap_remap = remap_using_cano(p)

    for atom in pmol.GetAtoms():
        xnum = atom.GetAtomMapNum()
        atom.SetAtomMapNum(amap_remap[xnum])

    r_update = []
    for reac in r.split('.'):
        mol = Chem.MolFromSmiles(reac)
        idx_amap = {x.GetIdx(): x.GetAtomMapNum() for x in mol.GetAtoms()}
        for atom in mol.GetAtoms():
            if atom.HasProp('molAtomMapNumber'):
                atom.ClearProp('molAtomMapNumber')
        ranks = list(Chem.CanonicalRankAtoms(mol))
        y = sorted(list(range(len(ranks))), key=lambda t: ranks[t])
        for t in y:
            xnum = idx_amap[t]
            if xnum not in amap_remap:
                amap_remap[xnum] = len(amap_remap) + 1
            atom = mol.GetAtomWithIdx(t)
            atom.SetAtomMapNum(amap_remap[xnum])
        r_update.append(Chem.MolToSmiles(mol))
    r_update = '.'.join(r_update)
    p_update = Chem.MolToSmiles(pmol)
    return f"{r_update}>>{p_update}"


def check_valid(rxn_smi):
    reac, prod = rxn_smi.split('>>')
    if reac == '' or prod == '':
        return False, "empty_mol"
    if '.' in prod:
        return False, 'multiple product moles'

    reac_mol = Chem.MolFromSmiles(reac)
    prod_mol = Chem.MolFromSmiles(prod)
    if reac_mol is None or prod_mol is None:
        return False, 'Invalid Smiles'

    prod_amap = [x.GetAtomMapNum() for x in prod_mol.GetAtoms()]
    pre_len = len(prod_amap)
    prod_amap = set(prod_amap)
    reac_amap = [x.GetAtomMapNum() for x in reac_mol.GetAtoms()]
    reac_amap = [x for x in reac_amap if x != 0]
    pre_reac_len = len(reac_amap)
    reac_amap = set(reac_amap)

    if 0 in prod_amap or len(prod_amap - reac_amap) > 0:
        return False, "Invalid atom mapping"

    if len(reac_amap) != pre_reac_len:
        return False, "Duplicate Amap in reac"

    if len(prod_amap) != pre_len:
        return False, "Duplicate Amap in prod"

    if len(prod_mol.GetAtoms()) == 1:
        return False, "Single Atom prod"

    return True, "correct"


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
        help="File with reactions to canonicalize"
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
                    reac, prod = reaction.split('>>')
                except Exception as e:
                    print('[Error Line]', lin)
                    print(e)
                    continue
                raw_len += 1
                is_valid, message = check_valid(rxn_smi=reaction)
                if not is_valid:
                    print('[reaction]', reaction)
                    print('[message]', message)
                    continue

                rxn_new = add_all_amap(reaction)
                rxn_new = clear_useless_part(reaction)
                rxn_new = remap_amap(rxn_new)
                id_marker += 1
                new_dict['id'].append(f'mit_{id_marker}')
                new_dict['class'].append(-1)
                new_dict['reactants>reagents>production'].append(rxn_new)

        new_df = pd.DataFrame.from_dict(new_dict)
        new_df.to_csv(outfile, index=False)
        print('[INFO] processed part:', part)
        print('[INFO] file size befoer process:', raw_len)
        print('[INFO] file size after process:', len(new_df))


if __name__ == "__main__":
    main()
