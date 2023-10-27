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
    'S_1_SP3': 1, 'O_0_SP3': 2, 'S_0_SP3D2': 3, 'Cu_0_SP3D2': 4, 'N_1_SP3': 5,
    'S_0_SP3D': 6, 'Br_0_SP3': 7, 'C_0_SP': 8, 'N_0_SP2': 9, 'S_0_SP3': 10,
    'P_0_SP3': 11, 'Sn_0_SP3': 12, 'I_0_SP3': 13, 'S_0_SP2': 14, 'C_0_SP3': 15,
    'P_0_SP2': 16, 'C_0_SP2': 17, 'S_-1_SP2': 18, 'C_-1_SP': 19, 'F_0_SP3': 20,
    'O_-1_SP3': 21, 'Mg_1_S': 22, 'Mg_0_SP': 23, 'N_-1_SP2': 24, 'O_-1_SP2': 25,
    'S_1_SP2': 26, 'Zn_0_SP': 27, 'Se_0_SP2': 28, 'Zn_1_S': 29, 'Si_0_SP3': 30,
    'N_0_SP': 31, 'N_1_SP2': 32, 'P_1_SP3': 33, 'P_0_SP3D': 34, 'O_0_SP2': 35,
    'N_1_SP': 36, 'S_-1_SP3': 37, 'Se_0_SP3': 38, 'Cl_0_SP3': 39, 'P_1_SP2': 40,
    'B_0_SP2': 41, 'N_0_SP3': 42
}

ATOM_IDX_TO_TYPE = {v: k for k, v in ATOM_TPYE_TO_IDX.items()}

ATOM_REMAP = {
    'B': 5, 'Br': 35, 'C': 6, 'Cl': 17, 'Cu': 29, 'F': 9, 'I': 53, 'Mg': 12,
    'N': 7, 'O': 8, 'P': 15, 'S': 16, 'Si': 14, 'Se': 34, 'Sn': 50, 'Zn': 30
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


def get_modified_atoms_bonds(
    reac: str, prod: str, kekulize: bool
) -> Tuple[Set, List]:
    reac_mol = get_mol(reac)
    prod_mol = get_mol(prod)

    if reac_mol is None or prod_mol is None:
        return set(), []
    if kekulize:
        reac_mol, prod_mol = align_kekule_pairs(reac, prod)

    prod_bonds = get_bond_info(prod_mol)
    prod_amap_idx = {
        atom.GetAtomMapNum(): atom.GetIdx()
        for atom in prod_mol.GetAtoms()
    }
    max_reac_amap = max(x.GetAtomMapNum() for x in reac_mol.GetAtoms())
    for atom in reac_mol.GetAtoms():
        if atom.GetAtomMapNum() == 0:
            atom.SetAtomMapNum(max_reac_amap + 1)
            max_reac_amap += 1

    reac_bonds = get_bond_info(reac_mol)
    reac_amap_idx = {
        atom.GetAtomMapNum(): atom.GetIdx()
        for atom in reac_mol.GetAtoms()
    }

    atom_edit, edge_edit = set(), []
    for bond in prod_bonds:
        if bond not in reac_bonds:
            edge_edit.append(bond)
            atom_edit.update(bond)
        else:
            reac_bond_type = reac_bonds[bond][0]
            prod_bond_type = prod_bonds[bond][0]
            if reac_bond_type != prod_bond_type:
                edge_edit.append(bond)
                atom_edit.update(bond)

    for bond in reac_bonds:
        if bond not in prod_bonds:
            start, end = bond
            if start in prod_amap_idx:
                atom_edit.add(start)
            if end in prod_amap_idx:
                atom_edit.add(end)

    for atom in prod_mol.GetAtoms():
        amap_num = atom.GetAtomMapNum()

        numHs_prod = atom.GetTotalNumHs()
        numHs_reac = reac_mol.GetAtomWithIdx(
            reac_amap_idx[amap_num]
        ).GetTotalNumHs()
        if numHs_prod != numHs_reac:
            atom_edit.add(amap_num)

    return atom_edit, edge_edit


def get_node_types(smiles, return_idx=True):
    mol = get_mol(smiles)
    result = {}
    for atom in mol.GetAtoms():
        amap_num = atom.GetAtomMapNum()
        hyb = atom.GetHybridization()
        sym = atom.GetSymbol()
        chg = atom.GetFormalCharge()
        result[amap_num] = f'{sym}_{chg}_{hyb}'

    if return_idx:
        result = {
            k: ATOM_TPYE_TO_IDX[v]
            for k, v in result.items()
        }

    return result


def get_edge_types(smiles, kekulize=False):
    mol = get_mol(smiles, kekulize=kekulize)
    result = {}
    for bond in mol.GetBonds():
        a_start = bond.GetBeginAtom().GetAtomMapNum()
        a_end = bond.GetEndAtom().GetAtomMapNum()
        bond_type = BOND_FLOAT_TO_IDX[bond.GetBondTypeAsDouble()]
        result[(a_start, a_end)] = bond_type
        result[(a_end, a_start)] = bond_type
    return result


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


def convert_res_into_smiles(
    org_node_types, org_edge_types, node_pred, edge_pred
):
    mol = Chem.RWMol()
    atom_reidx = {}
    for k, v in org_node_types.items():
        symbol, charge, _ = ATOM_IDX_TO_TYPE[v].split('_')
        new_idx = mol.AddAtom(Chem.Atom(ATOM_REMAP[symbol]))
        atom_reidx[k] = new_idx
        this_atom = mol.GetAtomWithIdx(new_idx)
        this_atom.SetFormalCharge(int(charge))

    for k, v in node_pred.items():
        symbol, charge, _ = ATOM_IDX_TO_TYPE[v].split('_')
        new_idx = mol.AddAtom(Chem.Atom(ATOM_REMAP[symbol]))
        atom_reidx[k] = new_idx
        this_atom = mol.GetAtomWithIdx(new_idx)
        this_atom.SetFormalCharge(int(charge))

    for (src, dst), v in org_edge_types.items():
        mol.AddBond(atom_reidx[src], atom_reidx[dst], BOND_TYPES[v])

    for (src, dst), v in edge_pred.items():
        mol.AddBond(atom_reidx[src], atom_reidx[dst], BOND_TYPES[v])

    mol = mol.GetMol()
    t_str = Chem.MolToSmiles(mol)
    return canonical_smiles(t_str)


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
