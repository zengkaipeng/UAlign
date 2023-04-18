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


def align_kekule_pairs(r: str, p: str) -> Tuple[Chem.Mol, Chem.Mol]:
    """Aligns kekule pairs to ensure unchanged bonds have same bond order in
    previously aromatic rings.
    Parameters
    ----------
    r: str,
        SMILES string representing the reactants
    p: str,
        SMILES string representing the product
    """
    reac_mol = Chem.MolFromSmiles(r)
    max_amap = max([atom.GetAtomMapNum() for atom in reac_mol.GetAtoms()])
    for atom in reac_mol.GetAtoms():
        if atom.GetAtomMapNum() == 0:
            atom.SetAtomMapNum(max_amap + 1)
            max_amap = max_amap + 1

    prod_mol = Chem.MolFromSmiles(p)

    prod_prev = get_bond_info(prod_mol)
    Chem.Kekulize(prod_mol)
    prod_new = get_bond_info(prod_mol)

    reac_prev = get_bond_info(reac_mol)
    Chem.Kekulize(reac_mol)
    reac_new = get_bond_info(reac_mol)

    for bond in prod_new:
        if bond in reac_new and (prod_prev[bond][0] == reac_prev[bond][0]):
            reac_new[bond][0] = prod_new[bond][0]

    reac_mol = Chem.RWMol(reac_mol)
    amap_idx = {
        atom.GetAtomMapNum(): atom.GetIdx()
        for atom in reac_mol.GetAtoms()
    }

    for bond in reac_new:
        idx1, idx2 = amap_idx[bond[0]], amap_idx[bond[1]]
        bo = reac_new[bond][0]
        reac_mol.RemoveBond(idx1, idx2)
        reac_mol.AddBond(idx1, idx2, BOND_FLOAT_TO_TYPE[bo])

    return reac_mol.GetMol(), prod_mol


def get_reaction_core(
    r: str, p: str, kekulize: bool = False,
) -> Tuple[Set, List]:
    """Get the reaction core and edits for given reaction
    Parameters
    ----------
    r: str,
        SMILES string representing the reactants
    p: str,
        SMILES string representing the product
    kekulize: bool,
        Whether to kekulize molecules to fetch minimal set of edits
    """
    reac_mol = get_mol(r)
    prod_mol = get_mol(p)

    if reac_mol is None or prod_mol is None:
        return set(), []

    if kekulize:
        reac_mol, prod_mol = align_kekule_pairs(r, p)

    prod_bonds = get_bond_info(prod_mol)
    p_amap_idx = {
        atom.GetAtomMapNum(): atom.GetIdx()
        for atom in prod_mol.GetAtoms()
    }

    max_amap = max([atom.GetAtomMapNum() for atom in reac_mol.GetAtoms()])
    for atom in reac_mol.GetAtoms():
        if atom.GetAtomMapNum() == 0:
            atom.SetAtomMapNum(max_amap + 1)
            max_amap += 1

    reac_bonds = get_bond_info(reac_mol)
    reac_amap = {
        atom.GetAtomMapNum(): atom.GetIdx()
        for atom in reac_mol.GetAtoms()
    }

    rxn_core = set()
    core_edits = []

    for bond in prod_bonds:
        if bond in reac_bonds and prod_bonds[bond][0] != reac_bonds[bond][0]:
            a_start, a_end = bond
            prod_bo, reac_bo = prod_bonds[bond][0], reac_bonds[bond][0]

            a_start, a_end = sorted([a_start, a_end])
            edit = f"{a_start}:{a_end}:{prod_bo}:{reac_bo}"
            core_edits.append(edit)
            rxn_core.update([a_start, a_end])

        if bond not in reac_bonds:
            a_start, a_end = bond
            reac_bo = 0.0
            prod_bo = prod_bonds[bond][0]

            start, end = sorted([a_start, a_end])
            edit = f"{a_start}:{a_end}:{prod_bo}:{reac_bo}"
            core_edits.append(edit)
            rxn_core.update([a_start, a_end])

    for bond in reac_bonds:
        if bond not in prod_bonds:
            amap1, amap2 = bond

            if (amap1 in p_amap_idx) and (amap2 in p_amap_idx):
                a_start, a_end = sorted([amap1, amap2])
                reac_bo = reac_bonds[bond][0]
                edit = f"{a_start}:{a_end}:{0.0}:{reac_bo}"
                core_edits.append(edit)
                rxn_core.update([a_start, a_end])

    for atom in prod_mol.GetAtoms():
        amap_num = atom.GetAtomMapNum()

        numHs_prod = atom.GetTotalNumHs()
        numHs_reac = reac_mol.GetAtomWithIdx(
            reac_amap[amap_num]
        ).GetTotalNumHs()

        if numHs_prod != numHs_reac:
            rxn_core.add(amap_num)

    return rxn_core, core_edits, reac_bonds


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
