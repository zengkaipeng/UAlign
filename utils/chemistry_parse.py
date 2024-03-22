import numpy as np
import re
from rdkit import Chem


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


def find_all_amap(smi):
    return list(map(int, re.findall(r"(?<=:)\d+", smi)))
