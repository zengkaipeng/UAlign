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


def augment_product_smiles(smi, do_random=False):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return smi

    frag_smis = Chem.MolToSmiles(mol).split('.')
    frags = []
    for frag_smi in frag_smis:
        frag_mol = Chem.MolFromSmiles(frag_smi)
        if frag_mol is None:
            return smi
        frags.append((
            frag_mol.GetNumAtoms(),
            Chem.MolToSmiles(frag_mol, doRandom=do_random),
        ))

    frags.sort(key=lambda item: (-item[0], item[1]))
    return '.'.join(frag_smi for _, frag_smi in frags)


def remove_am_wo_cano(smi):
    mol = Chem.MolFromSmiles(smi)
    for atom in mol.GetAtoms():
        if atom.HasProp('molAtomMapNumber'):
            atom.ClearProp('molAtomMapNumber')

    return Chem.MolToSmiles(mol, canonical=False)


def find_all_amap(smi):
    return list(map(int, re.findall(r"(?<=:)\d+", smi)))
