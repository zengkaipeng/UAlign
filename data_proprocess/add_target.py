import os
from tqdm import tqdm
import argparse
import multiprocessing
import rdkit
from rdkit import Chem
import pandas


def clear_map_number(smi):
    """Clear the atom mapping number of a SMILES sequence"""
    mol = Chem.MolFromSmiles(smi)
    for atom in mol.GetAtoms():
        if atom.HasProp('molAtomMapNumber'):
            atom.ClearProp('molAtomMapNumber')
    return canonical_smiles(Chem.MolToSmiles(mol))


def canonical_smiles(smi):
    """Canonicalize a SMILES without atom mapping"""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return smi
    else:
        canonical_smi = Chem.MolToSmiles(mol)
        # print('>>', canonical_smi)
        if '.' in canonical_smi:
            canonical_smi_list = canonical_smi.split('.')
            canonical_smi_list = sorted(
                canonical_smi_list, key=lambda x: (len(x), x)
            )
            canonical_smi = '.'.join(canonical_smi_list)
        return canonical_smi


def clear_batch(pid, x):
    result = [clear_map_number(t) for t in tqdm(x, desc=f'Part {pid}')]
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--file_dir', required=True, type=str,
        help='the dir containing canonicalized files'
    )
    parser.add_argument(
        '--num_proc', type=int, default=-1,
        help='the number of process to process data'
    )
    parser.add_argument(
        '--batch_size', type=int, default=10000,
        help='the data size for a sub-progress to process'
    )
    args = parser.parse_args()

    part2input = {
        'train': 'canonicalized_raw_train.csv',
        'valid': 'canonicalized_raw_val.csv',
        'test': 'canonicalized_raw_test.csv'
    }

    part2output = {
        'train': 'extend_train.csv',
        'valid': 'extend_valid.csv',
        'test': 'extend_test.csv'
    }

    NPROC = min(16, os.cpu_count() - 2) \
        if args.num_proc == -1 else args.num_proc

    for part in ['train', 'valid', 'test']:
        meta = pandas.read_csv(os.path.join(args.file_dir, part2input[part]))
        ret = [x.split('>>')[0] for x in meta['reactants>reagents>production']]
        pol, res = multiprocessing.Pool(processes=NPROC), []
        for idx in range(0, len(ret), args.batch_size):
            res.append(pol.apply_async(
                clear_batch, args=(
                    f'{part}_{idx}/{len(ret)}',
                    ret[idx: idx + args.batch_size]
                )
            ))

        pol.close()
        pol.join()

        all_clean_ret = []
        for x in res:
            all_clean_ret.extend(x.get())

        meta.insert(3, 'clean_reactant', all_clean_ret)
        meta.to_csv(os.path.join(args.file_dir, part2output[part]))
