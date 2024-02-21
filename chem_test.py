from rdkit import Chem
from copy import deepcopy
import numpy as np

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


def find_all_amap(smi):
    return list(map(int, re.findall(r"(?<=:)\d+", smi)))


def cano_with_am(smi):
    mol = Chem.MolFromSmiles(smi)
    tmol = deepcopy(mol)
    for atom in mol.GetAtoms():
        if atom.HasProp('molAtomMapNumber'):
            atom.ClearProp('molAtomMapNumber')

    ranks = list(Chem.CanonicalRankAtoms(mol))
    root_atom = int(np.argmin(ranks))
    return Chem.MolToSmiles(tmol, rootedAtAtom=root_atom, canonical=True)


def remove_am_wo_cano(smi):
    mol = Chem.MolFromSmiles(smi)
    for atom in mol.GetAtoms():
        if atom.HasProp('molAtomMapNumber'):
            atom.ClearProp('molAtomMapNumber')

    return Chem.MolToSmiles(mol, canonical=False)


def clear_map_number(smi):
    """Clear the atom mapping number of a SMILES sequence"""
    mol = Chem.MolFromSmiles(smi)
    for atom in mol.GetAtoms():
        if atom.HasProp('molAtomMapNumber'):
            atom.ClearProp('molAtomMapNumber')
    return canonical_smiles(Chem.MolToSmiles(mol))


x = '[CH3:1][C:2]([CH3:3])([CH3:4])[O:5][C:6](=[O:7])[n:8]1[c:9]2[cH:10][cH:11][c:12]([C:13]([CH3:14])=[O:15])[cH:16][c:17]2[cH:18][cH:19]1'

cano_ranks = get_cano_ams(x)

old_rank_to_new = {v: idx + 1 for idx, v in enumerate(cano_ranks)}

mol = Chem.MolFromSmiles(x)

for t in mol.GetAtoms():
    t.SetAtomMapNum(old_rank_to_new[t.GetAtomMapNum()])

y = Chem.MolToSmiles(mol)

print(cano_with_am(x))
print(cano_with_am(y))
print(clear_map_number(x))
