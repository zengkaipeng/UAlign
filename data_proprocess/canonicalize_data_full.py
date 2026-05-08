"""
Canonicalize the product SMILES, and then use substructure matching to infer
the correspondence to the original atom-mapped order. This correspondence is then
used to renumber the reactant atoms.
"""

import rdkit
import os
import argparse
from multiprocessing import Pool
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


def process_reaction(row):
    uspto_id, rxn_smi = row
    is_valid, message = check_valid(rxn_smi)
    if not is_valid:
        return None, rxn_smi, message

    rxn_new = add_all_amap(rxn_smi)
    rxn_new = remap_amap(rxn_new)
    return {
        'id': uspto_id,
        'class': -1,
        'reactants>reagents>production': rxn_new,
    }, None, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--filename", required=True,
        help="File with reactions to canonicalize"
    )
    parser.add_argument(
        "--num_procs", type=int, default=4,
        help="number of processes for canonicalization"
    )
    parser.add_argument(
        "--chunksize", type=int, default=128,
        help="number of reactions assigned to each worker task"
    )
    args = parser.parse_args()

    file_path = os.path.abspath(args.filename)
    file_name = os.path.basename(file_path)
    file_dir = os.path.dirname(file_path)
    new_file = f"canonicalized_{file_name}"
    df = pd.read_csv(args.filename)
    print(f"Processing file of size: {len(df)}")

    rows = list(
        df[['id', 'reactants>reagents>production']].itertuples(
            index=False, name=None
        )
    )
    num_procs = max(1, args.num_procs)
    if num_procs == 1:
        result_iter = map(process_reaction, rows)
        progress_iter = tqdm(result_iter, total=len(rows))
        records = []
        for record, rxn_smi, message in progress_iter:
            if record is None:
                print('[reaction]', rxn_smi)
                print('[message]', message)
                continue
            records.append(record)
    else:
        with Pool(processes=num_procs) as pool:
            progress_iter = tqdm(
                pool.imap(
                    process_reaction, rows, chunksize=max(1, args.chunksize)
                ),
                total=len(rows)
            )
            records = []
            for record, rxn_smi, message in progress_iter:
                if record is None:
                    print('[reaction]', rxn_smi)
                    print('[message]', message)
                    continue
                records.append(record)

    new_df = pd.DataFrame.from_records(
        records,
        columns=['id', 'class', 'reactants>reagents>production']
    )
    new_df.to_csv(f"{file_dir}/{new_file}", index=False)

    print('[INFO] file size after process:', len(new_df))


if __name__ == "__main__":
    main()
