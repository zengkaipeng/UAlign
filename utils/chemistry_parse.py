import numpy as np
import re
from copy import deepcopy
from rdkit import Chem
from typing import List, Dict, Set, Tuple
from rdkit.Chem import AllChem
from rdkit.Chem import Draw


BOND_TYPES = [
    None, Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC
]
BOND_FLOAT_TO_TYPE = {
    0.0: BOND_TYPES[0],
    1.0: BOND_TYPES[1],
    2.0: BOND_TYPES[2],
    3.0: BOND_TYPES[3],
    1.5: BOND_TYPES[4],
}

BOND_FLOAT_TO_IDX = {0.0: 0, 1.0: 1, 2.0: 2, 3.0: 3, 1.5: 4}

ATOM_TPYE_TO_IDX = {
    'Zn_1': 1, 'S_-1': 2, 'Mg_1': 3, 'C_0_SP2': 4, 'Si_0': 5,
    'S_0': 6, 'Mg_0': 7, 'N_1': 8, 'Cu_0': 9, 'Zn_0': 10, 'P_1': 11,
    'O_0': 12, 'O_-1': 13, 'C_-1_SP': 14, 'S_1': 15, 'Br_0': 16, 'P_0': 17,
    'C_0_SP': 18, 'Sn_0': 19, 'B_0': 20, 'Se_0': 21, 'F_0': 22, 'I_0': 23,
    'N_-1': 24, 'N_0': 25, 'C_0_SP3': 26, 'Cl_0': 27
}

ATOM_IDX_TO_TYPE = {v: k for k, v in ATOM_TPYE_TO_IDX.items()}

ATOM_REMAP = {
    'B': 5, 'Br': 35, 'C': 6, 'Cl': 17, 'Cu': 29, 'F': 9, 'I': 53, 'Mg': 12,
    'N': 7, 'O': 8, 'P': 15, 'S': 16, 'Si': 14, 'Se': 34, 'Sn': 50, 'Zn': 30
}


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


def get_bond_info(mol: Chem.Mol) -> Dict:
    """Get information on bonds in the molecule.
    Parameters
    ----------
    mol: Chem.Mol
        Molecule
    """
    if mol is None:
        return {}

    bond_info = {}
    for bond in mol.GetBonds():
        a_start = bond.GetBeginAtom().GetAtomMapNum()
        a_end = bond.GetEndAtom().GetAtomMapNum()

        key_pair = sorted([a_start, a_end])
        bond_info[tuple(key_pair)] = [
            bond.GetBondTypeAsDouble(), bond.GetIdx()
        ]

    return bond_info


def get_mol(smiles: str, kekulize: bool = False) -> Chem.Mol:
    """SMILES string to Mol.
    Parameters
    ----------
    smiles: str,
        SMILES string for molecule
    kekulize: bool,
        Whether to kekulize the molecule
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None and kekulize:
        Chem.Kekulize(mol)
    return mol


def get_synthon_edits(
    reac: str, prod: str, consider_inner_bonds: bool = False,
    return_org_type: bool = False
):
    reac_mol, prod_mol = get_mol(reac), get_mol(prod)
    ke_reac_mol = get_mol(reac, kekulize=True)
    if reac_mol is None or prod_mol is None:
        raise NotImplementedError('[SYN BREAK] Invalid Smiles Given')

    prod_bonds = get_bond_info(prod_mol)
    reac_bonds = get_bond_info(reac_mol)

    prod_amap_idx = {
        atom.GetAtomMapNum(): atom.GetIdx()
        for atom in prod_mol.GetAtoms()
    }

    reac_amap_idx = {
        atom.GetAtomMapNum(): atom.GetIdx()
        for atom in reac_mol.GetAtoms()
    }
    ke_reac_bonds = get_bond_info(ke_reac_mol)

    Eatom, Hatom, Catom, deltaE = set(), set(), set(), {}
    for bond, (ftype, bidx) in prod_bonds.items():
        if bond not in reac_bonds:
            target = 0
        elif ftype == reac_bonds[bond][0]:
            continue
        elif reac_bonds[bond][0] == 1.5:
            target = ke_reac_bonds[bond][0]
        else:
            target = reac_bonds[bond][0]

        assert target != 1.5, 'Building aromatic bonds!'
        if ftype != target:
            deltaE[bond] = (ftype, target)
            Eatom.update(bond)

    if consider_inner_bonds:
        for bond in reac_bonds:
            if bond[0] not in prod_amap_idx or bond[1] not in prod_amap_idx:
                continue
            if bond not in prod_bonds:
                deltaE[bond] = (0, ke_reac_bonds[bond][0])
                Eatom.update(bond)

    for atom in prod_mol.GetAtoms():
        amap_num = atom.GetAtomMapNum()
        reac_atom = reac_mol.GetAtomWithIdx(reac_amap_idx[amap_num])
        if atom.GetTotalNumHs() != reac_atom.GetTotalNumHs():
            Hatom.add(amap_num)
        if atom.GetFormalCharge() != reac_atom.GetFormalCharge():
            Catom.add(amap_num)

    if not return_org_type:
        return Eatom, Hatom, Catom, deltaE
    return Eatom, Hatom, Catom, deltaE, prod_bonds


if __name__ == '__main__':
    with open('test_examples.txt') as Fin:
        content = Fin.readlines()

    # example I
    reac, prod = content[0].strip().split('>>')
    mols = [Chem.MolFromSmiles(reac), Chem.MolFromSmiles(prod)]
    img = Draw.MolsToGridImage(
        mols,
        molsPerRow=2,
        subImgSize=(2000, 2000),
        legends=["REAC", 'PROD']
    )
    img.save('tmp_figs/example1.pdf')

    rxn_core, core_edits, real_bond = get_reaction_core(reac, prod, True, True)
    print('[RXN CORE]')
    print(rxn_core)
    print('[CORE EDIT]')
    print(core_edits)

    # example 2

    reac, prod = content[1].strip().split('>>')
    mols = [Chem.MolFromSmiles(reac), Chem.MolFromSmiles(prod)]
    img = Draw.MolsToGridImage(
        mols,
        molsPerRow=2,
        subImgSize=(2000, 2000),
        legends=["REAC", 'PROD']
    )
    img.save('tmp_figs/example2.pdf')

    p_mol = mols[1]
    print([x.GetSymbol() for x in p_mol.GetAtoms()])
    p_mol = Chem.AddHs(p_mol)
    print([x.GetSymbol() for x in p_mol.GetAtoms()])


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
