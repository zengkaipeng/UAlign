"""
Shared CSV canonicalization for USPTO-50K and USPTO-FULL.

The chemistry preprocessing is identical for both datasets; the only dataset
specific behavior is how the output reaction class is assigned.
"""

import argparse
import os
from multiprocessing import Pool

import pandas as pd
from rdkit import Chem
from tqdm import tqdm


REACTION_COLUMN = 'reactants>reagents>production'


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


def check_valid(rxn_smi, reject_multi_product=False):
    reac, prod = rxn_smi.split('>>')
    if reac == '' or prod == '':
        return False, "empty_mol"
    if reject_multi_product and '.' in prod:
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


def process_reaction(row):
    uspto_id, class_id, rxn_smi = row
    is_valid, message = check_valid(rxn_smi)
    if not is_valid:
        return None, rxn_smi, message

    rxn_new = add_all_amap(rxn_smi)
    rxn_new = remap_amap(rxn_new)
    return {
        'id': uspto_id,
        'class': class_id,
        REACTION_COLUMN: rxn_new,
    }, None, None


def build_rows(df, class_mode):
    if 'id' not in df.columns or REACTION_COLUMN not in df.columns:
        raise ValueError(f"Input CSV must contain 'id' and '{REACTION_COLUMN}'")

    if class_mode == 'preserve':
        if 'class' not in df.columns:
            raise ValueError("class_mode='preserve' requires a 'class' column")
        return list(df[['id', 'class', REACTION_COLUMN]].itertuples(
            index=False, name=None
        ))

    if class_mode == 'minus_one':
        return [
            (uspto_id, -1, rxn_smi)
            for uspto_id, rxn_smi in df[['id', REACTION_COLUMN]].itertuples(
                index=False, name=None
            )
        ]

    raise ValueError(f"Unsupported class_mode: {class_mode}")


def iter_results(rows, num_procs, chunksize):
    if num_procs == 1:
        return tqdm(map(process_reaction, rows), total=len(rows))

    pool = Pool(processes=num_procs)
    return pool, tqdm(
        pool.imap(process_reaction, rows, chunksize=chunksize),
        total=len(rows)
    )


def canonicalize_file(filename, class_mode, num_procs=4, chunksize=128):
    file_path = os.path.abspath(filename)
    file_name = os.path.basename(file_path)
    file_dir = os.path.dirname(file_path)
    new_file = f"canonicalized_{file_name}"
    df = pd.read_csv(filename)
    print(f"Processing file of size: {len(df)}")
    print(f"[INFO] class mode: {class_mode}")

    rows = build_rows(df, class_mode)
    num_procs = max(1, num_procs)
    chunksize = max(1, chunksize)
    records = []

    if num_procs == 1:
        result_iter = iter_results(rows, num_procs, chunksize)
        for record, rxn_smi, message in result_iter:
            if record is None:
                print('[reaction]', rxn_smi)
                print('[message]', message)
                continue
            records.append(record)
    else:
        pool, result_iter = iter_results(rows, num_procs, chunksize)
        with pool:
            for record, rxn_smi, message in result_iter:
                if record is None:
                    print('[reaction]', rxn_smi)
                    print('[message]', message)
                    continue
                records.append(record)

    new_df = pd.DataFrame.from_records(
        records,
        columns=['id', 'class', REACTION_COLUMN]
    )
    new_df.to_csv(f"{file_dir}/{new_file}", index=False)

    print('[INFO] file size after process:', len(new_df))


def build_parser(
    description='Canonicalize USPTO CSV data',
    class_mode=None,
    include_class_mode=True,
    default_num_procs=4,
):
    parser = argparse.ArgumentParser(description)
    parser.add_argument(
        "--filename", required=True,
        help="File with reactions to canonicalize"
    )
    if include_class_mode:
        class_mode_kwargs = {
            'choices': ['preserve', 'minus_one'],
            'help': (
                "preserve the input class column or write -1 for all classes"
            ),
        }
        if class_mode is None:
            class_mode_kwargs['required'] = True
        else:
            class_mode_kwargs['default'] = class_mode
        parser.add_argument("--class_mode", **class_mode_kwargs)
    parser.add_argument(
        "--num_procs", type=int, default=default_num_procs,
        help="number of processes for canonicalization"
    )
    parser.add_argument(
        "--chunksize", type=int, default=128,
        help="number of reactions assigned to each worker task"
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    canonicalize_file(
        args.filename,
        class_mode=args.class_mode,
        num_procs=args.num_procs,
        chunksize=args.chunksize,
    )


if __name__ == "__main__":
    main()
