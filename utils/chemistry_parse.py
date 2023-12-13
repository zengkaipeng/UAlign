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
BOND_FLOAT_TYPES = [0.0, 1.0, 2.0, 3.0, 1.5]


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

ACHANGE_TO_IDX = {0: 0, 1: 1, 2: 2, 3: 3, -1: 4, -2: 5, - 3: 6}


def get_all_amap(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return set()
    answer = set(x.GetAtomMapNum() for x in mol.GetAtoms())
    return answer


def get_mol_belong(smi, belong):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        raise NotImplementedError(f'Invalid smiles {smi}')
    for atom in mol.GetAtoms():
        return belong[atom.GetAtomMapNum()]
    return -1


def clear_map_number(smi):
    """Clear the atom mapping number of a SMILES sequence"""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        print(smi)
        return smi
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
                canonical_smi_list, key=lambda x: (len(x), x))
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
    r: str, p: str, kekulize: bool = False
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
            edit = f"{amap_num}:{0}:{1.0}:{0.0}"
            core_edits.append(edit)
            rxn_core.add(amap_num)

    return rxn_core, core_edits


def break_fragements(smiles, break_edges, canonicalize=False):
    """ 

    break a smilse into synthons according to the given break edges
    the break edges is a Iterable of tuples. tuple contains the amap 
    numbers of end atoms.
    """

    Mol = Chem.MolFromSmiles(smiles)
    Chem.Kekulize(Mol, True)
    # The kekulize is required otherwise you can not get the
    # correct synthons

    assert all(x.GetAtomMapNum() != 0 for x in Mol.GetAtoms()), \
        'Invalid atom mapping is founded, please correct it'

    tmol = Chem.RWMol(Mol)

    for bond in Mol.GetBonds():
        start_atom = bond.GetBeginAtom()
        end_atom = bond.GetEndAtom()
        start_idx = start_atom.GetIdx()
        end_idx = end_atom.GetIdx()
        start_amap = start_atom.GetAtomMapNum()
        end_amap = end_atom.GetAtomMapNum()

        if (start_amap, end_amap) in break_edges:
            tmol.RemoveBond(start_idx, end_idx)

    answer = Chem.MolToSmiles(tmol.GetMol())
    if Chem.MolFromSmiles(answer) is None:
        print('\n[smi]', smiles)
    if canonicalize:
        answer = clear_map_number(answer)
    return answer


def get_leaving_group_synthon(
    prod: str, reac: str, consider_inner_bonds: bool = False
) -> Tuple[List[str], List[str], Dict[Tuple[int, int], float]]:
    prod_amap = get_all_amap(prod)
    reac_amap = get_all_amap(reac)

    p_mol = get_mol(prod, kekulize=True)
    r_mol = get_mol(reac, kekulize=True)

    if p_mol is None or r_mol is None:
        raise NotImplementedError('[LG EXT] Invalid Smiles Given')

    prod_bonds = set()

    for bond in p_mol.GetBonds():
        start_atom = bond.GetBeginAtom()
        end_atom = bond.GetEndAtom()
        start_amap = start_atom.GetAtomMapNum()
        end_amap = end_atom.GetAtomMapNum()
        prod_bonds.add((start_amap, end_amap))
        prod_bonds.add((end_amap, start_amap))

    break_edges, conn_edges = set(), {}
    for bond in r_mol.GetBonds():
        start_atom = bond.GetBeginAtom()
        end_atom = bond.GetEndAtom()
        start_amap = start_atom.GetAtomMapNum()
        end_amap = end_atom.GetAtomMapNum()
        e_type = bond.GetBondTypeAsDouble()

        if start_amap in prod_amap:
            if end_amap not in prod_amap:
                break_edges.add((start_amap, end_amap))
                break_edges.add((end_amap, start_amap))
                conn_edges[(start_amap, end_amap)] = e_type
            elif not consider_inner_bonds and \
                    (start_amap, end_amap) not in prod_amap:
                break_edges.add((start_amap, end_amap))
                break_edges.add((end_amap, start_amap))

        elif end_amap in prod_amap:
            break_edges.add((start_amap, end_amap))
            break_edges.add((end_amap, start_amap))
            conn_edges[(end_amap, start_amap)] = e_type

    frgs = break_fragements(reac, break_edges).split('.')
    lgs, syns = [], []
    for block in frgs:
        all_amap = get_all_amap(block)
        if len(all_amap & prod_amap) == 0:
            lgs.append(block)
        else:
            assert len(all_amap & prod_amap) == len(all_amap), \
                f'The breaking is not correct, {reac}>>{prod}'
            syns.append(block)

    return lgs, syns, conn_edges


def get_synthon_breaks(reac: str, prod: str):
    reac_mol, prod_mol = get_mol(reac), get_mol(prod)
    if reac_mol is None or prod_mol is None:
        raise NotImplementedError('[SYN BREAK] Invalid Smiles Given')

    prod_bonds = get_bond_info(prod_mol)
    reac_bonds = get_bond_info(reac_mol)
    break_edges = set(prod_bonds.keys()) - set(reac_bonds.keys())
    modified_atoms = set()
    for a, b in break_edges:
        modified_atoms.update((a, b))
    return modified_atoms, break_edges


def get_synthon_edits(
    broken_reac: str, broken_prod: str, consider_inner_bonds: bool = False
) -> Set[int], Dict[Tuple[int, int], float]:
    reac_mol, prod_mol = get_mol(broken_reac), get_mol(broken_prod)
    if reac_mol is None or prod_mol is None:
        raise NotImplementedError('[SYN EDIT] Invalid Smiles Given')
    ke_reac_mol = get_mol(broken_reac, kekulize=True)
    ke_reac_bonds = get_bond_info(ke_reac_mol)

    prod_bonds = get_bond_info(prod_mol)
    prod_amap_idx = {
        atom.GetAtomMapNum(): atom.GetIdx()
        for atom in prod_mol.GetAtoms()
    }

    reac_bonds = get_bond_info(reac_mol)
    reac_amap_idx = {
        atom.GetAtomMapNum(): atom.GetIdx()
        for atom in reac_mol.GetAtoms()
    }

    modified_atoms, deltaE = set(), {}
    for bond in prod_bonds:
        target = reac_bonds[bond][0]
        if prod_bonds[bond][0] == target:
            continue
        if target == 1.5:
            target = ke_reac_bonds[bond][0]
        deltaE[bond] = (prod_bonds[bond][0], reac_bonds[bond][0])
        modified_atoms.update(bond)

    if consider_inner_bonds:
        for bond in reac_bonds:
            if bond not in prod_bonds:
                deltaE[bond] = (0, reac_bonds[bond][0])
                modified_atoms.update(bond)

    for atom in prod_mol.GetAtoms():
        amap_num = atom.GetAtomMapNum()
        numHs_prod = atom.GetTotalNumHs()
        reac_atom = reac_mol.GetAtomWithIdx(reac_amap_idx[amap_num])
        numHs_reac = reac_atom.GetTotalNumHs()
        if numHs_reac != numHs_prod:
            modified_atoms.add(amap_num)

    return modified_atoms, deltaE


def edit_to_synthons(smi, break_edge, edge_edits):
    mol = Chem.MolFromSmiles(smi)
    Chem.Kekulize(mol, True)
    # clear all the aromatic information, to avoid invalid breaking

    tmol = Chem.RWMol(mol)
    amap = {x.GetAtomMapNum(): x.GetIdx() for x in tmol.GetAtoms()}
    for (a, b) in break_edges:
        a_idx, b_idx = amap[a], amap[b]
        begin_atom = tmol.GetAtomWithIdx(a_idx)
        end_atom = tmol.GetAtomWithIdx(b_idx)
        a_sym, b_symb = begin_atom.GetSymbol(), end_atom.GetSymbol()
        old_bond = tmol.GetBondBetweenAtoms(a_idx, b_idx)
        if old_bond is not None:
            tmol.RemoveBond(a_idx, b_idx)
            old_type = old_bond.GetBondTypeAsDouble()
            if old_type == 1:
                if a_sym == 'N' and b_sym == 'O':
                    if begin_atom.GetFormalCharge() == 1:
                        begin_atom.SetFormalCharge(0)
                        begin_atom.SetNumExplicitHs(0)
                    if end_atom.GetFormalCharge() == -1:
                        end_atom.SetFormalCharge(0)

                if a_sym == 'O' and b_sym == 'N':
                    if begin_atom.GetFormalCharge() == -1:
                        begin_atom.SetFormalCharge(0)
                    if end_atom.GetFormalCharge() == 1:
                        end_atom.SetFormalCharge(0)
                        end_atom.SetNumExplicitHs(0)

    # break first
    break_mol = tmol.GetMol()
    break_smi = Chem.MolToSmiles(break_mol)

    broken_mol = Chem.MolFromSmiles(break_smi)
    tmol = Chem.RWMol(broken_mol)

    amap = {x.GetAtomMapNum(): x.GetIdx() for x in tmol.GetAtoms()}
    for (a_amap, b_amap), new_type in edge_edits.items():
        a_idx, b_idx = amap[a], amap[b]
        begin_atom = tmol.GetAtomWithIdx(a_idx)
        end_atom = tmol.GetAtomWithIdx(b_idx)
        a_sym, b_symb = begin_atom.GetSymbol(), end_atom.GetSymbol()
        old_bond = tmol.GetBondBetweenAtoms(a_idx, b_idx)
        old_type = 0 if old_bond is None else old_bond.GetBondTypeAsDouble()
        if old_type != new_type:
            tmol.RemoveBond(a_idx, b_idx)
            tmol.AddBond(a_idx, b_idx, BOND_FLOAT_TO_TYPE[new_type])

            if old_type == 1 and new_type == 2:
                if a_sym == 'S' and b_sym == 'O' and \
                        end_atom.GetFormalCharge() == -1:
                    end_atom.SetFormalCharge(0)
                if a_sym == 'O' and b_sym == 'S' and \
                        begin_atom.GetFormalCharge() == -1:
                    begin_atom.SetFormalCharge(0)

            elif new_type > old_type:
                a_hs = begin_atom.GetNumExplicitHs()
                b_hs = end_atom.GetNumExplicitHs()
                delta = new_type - ke_old_bond
                begin_atom.SetNumExplicitHs(int(max(0, a_hs - delta)))
                end_atom.SetNumExplicitHs(int(max(0, b_hs - delta)))
    syn_mol = tmol.GetMol()
    return Chem.MolToSmiles(syn_mol)


def edit_to_synthons(smi, edge_types):
    mol = Chem.MolFromSmiles(smi)
    Chem.Kekulize(mol, True)
    # clear all the aromatic information, to avoid invalid breaking

    tmol = Chem.RWMol(mol)
    amap = {x.GetAtomMapNum(): x.GetIdx() for x in tmol.GetAtoms()}
    bonds = []
    for bond in mol.GetBonds():
        begin_atom, end_atom = bond.GetBeginAtom(), bond.GetEndAtom()
        a_amap, b_amap = begin_atom.GetAtomMapNum(), end_atom.GetAtomMapNum()
        key_pair = (min(a_amap, b_amap), max(a_amap, b_amap))
        bonds.append(key_pair)

    for (a_amap, b_amap) in bonds:
        a_idx, b_idx = amap[a_amap], amap[b_amap]
        begin_atom = tmol.GetAtomWithIdx(a_idx)
        end_atom = tmol.GetAtomWithIdx(b_idx)
        a_sym, b_sym = begin_atom.GetSymbol(), end_atom.GetSymbol()

        old_type, new_type = edge_types[(a_amap, b_amap)]
        ke_bond = tmol.GetBondBetweenAtoms(a_idx, b_idx)
        ke_old_bond = 0 if ke_bond is None else ke_bond.GetBondTypeAsDouble()

        if old_type == new_type or ke_old_bond == new_type:
            # bond types will change after removing aromatic info
            continue
        if new_type == 1.5:
            raise NotImplementedError('Building aromatic edges forbidden')

        if new_type == 0:
            tmol.RemoveBond(a_idx, b_idx)
            if old_type == 1:
                if a_sym == 'N' and b_sym == 'O':
                    if begin_atom.GetFormalCharge() == 1:
                        begin_atom.SetFormalCharge(0)
                        begin_atom.SetNumExplicitHs(0)
                    if end_atom.GetFormalCharge() == -1:
                        end_atom.SetFormalCharge(0)

                if a_sym == 'O' and b_sym == 'N':
                    if begin_atom.GetFormalCharge() == -1:
                        begin_atom.SetFormalCharge(0)
                    if end_atom.GetFormalCharge() == 1:
                        end_atom.SetFormalCharge(0)
                        end_atom.SetNumExplicitHs(0)

            # we do not add Hs on it because this is to get synthon

        else:
            tmol.RemoveBond(a_idx, b_idx)
            tmol.AddBond(a_idx, b_idx, BOND_FLOAT_TO_TYPE[new_type])
            if old_type == 1 and new_type == 2:
                if a_sym == 'S' and b_sym == 'O' and \
                        end_atom.GetFormalCharge() == -1:
                    end_atom.SetFormalCharge(0)
                if a_sym == 'O' and b_sym == 'S' and \
                        begin_atom.GetFormalCharge() == -1:
                    begin_atom.SetFormalCharge(0)

            if new_type > ke_old_bond:
                a_hs = begin_atom.GetNumExplicitHs()
                b_hs = end_atom.GetNumExplicitHs()
                delta = new_type - ke_old_bond
                assert int(delta) == delta, "aromatic info not remove"
                begin_atom.SetNumExplicitHs(int(max(0, a_hs - delta)))
                end_atom.SetNumExplicitHs(int(max(0, b_hs - delta)))

            # preserve the Hs un added because this is to get synthon

    for atom in tmol.GetAtoms():
        if atom.GetSymbol() == 'N':
            bond_vals = sum([x.GetBondTypeAsDouble() for x in atom.GetBonds()])
            if bond_vals == 4:
                atom.SetFormalCharge(1)
        elif atom.GetSymbol() == 'P':
            bond_vals = [x.GetBondTypeAsDouble() for x in atom.GetBonds()]
            if sum(bond_vals) == 4 and len(bond_vals) == 4:
                atom.SetFormalCharge(1)
                atom.SetNumExplicitHs(0)

    new_mol = tmol.GetMol()
    answer = Chem.MolToSmiles(new_mol)
    if Chem.MolFromSmiles(answer) is None:
        print('[smi]', smi)
        print('[ans]', answer)
    return answer


# def get_synthons(prod: str, reac: str, kekulize: bool = False):
#     reac_mol, prod_mol = get_mol(reac), get_mol(prod)
#     if reac_mol is None or prod_mol is None:
#         return {}, {}
#     if kekulize:
#         reac_mol, prod_mol = align_kekule_pairs(reac, prod)
#     ke_reac_mol = get_mol(reac, kekulize=True)
#     ke_reac_bonds = get_bond_info(ke_reac_mol)

#     prod_bonds = get_bond_info(prod_mol)
#     prod_amap_idx = {
#         atom.GetAtomMapNum(): atom.GetIdx()
#         for atom in prod_mol.GetAtoms()
#     }

#     reac_bonds = get_bond_info(reac_mol)
#     reac_amap_idx = {
#         atom.GetAtomMapNum(): atom.GetIdx()
#         for atom in reac_mol.GetAtoms()
#     }

#     atom2deltaH, edges2typechange = {}, {}

#     for bond in prod_bonds:
#         target_type = reac_bonds[bond][0] if bond in reac_bonds else 0.0
#         if prod_bonds[bond][0] != 1.5 and target_type == 1.5:
#             target_type = ke_reac_bonds[bond][0]
#         # when there is an aromatic bond of reactants breaks
#         # we choose to recover it using only single or double bond
#         # instead of aromatic bond to make the code short
#         edges2typechange[bond] = (prod_bonds[bond][0], target_type)

#     for atom in prod_mol.GetAtoms():
#         amap_num = atom.GetAtomMapNum()
#         numHs_prod = atom.GetTotalNumHs()
#         reac_atom = reac_mol.GetAtomWithIdx(reac_amap_idx[amap_num])
#         numHs_reac = reac_atom.GetTotalNumHs()
#         atom2deltaH[amap_num] = numHs_prod - numHs_reac

#     # We have omitted the formation of new bonds between the product atoms
#     # during the reaction process, as this situation occurs
#     # only to a small extent.

#     return atom2deltaH, edges2typechange


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
        symbol, charge = ATOM_IDX_TO_TYPE[v].split('_')[:2]
        new_idx = mol.AddAtom(Chem.Atom(ATOM_REMAP[symbol]))
        atom_reidx[k] = new_idx
        this_atom = mol.GetAtomWithIdx(new_idx)
        this_atom.SetFormalCharge(int(charge))

    for k, v in node_pred.items():
        symbol, charge = ATOM_IDX_TO_TYPE[v].split('_')[:2]
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


def add_random_Amap(smiles):
    mol = Chem.MolFromSmiles(smiles)
    for x in mol.GetAtoms():
        x.ClearProp('molAtomMapNumber')
    for idx, x in enumerate(mol.GetAtoms()):
        x.SetAtomMapNum(idx + 1)
    return Chem.MolToSmiles(mol)


def get_node_types(smiles, return_idx=True):
    mol = get_mol(smiles)
    result = {}
    for atom in mol.GetAtoms():
        amap_num = atom.GetAtomMapNum()
        hyb = atom.GetHybridization()
        sym = atom.GetSymbol()
        chg = atom.GetFormalCharge()
        if sym == 'C':
            result[amap_num] = f'{sym}_{chg}_{hyb}'
        else:
            result[amap_num] = f'{sym}_{chg}'

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


def get_block(reactants):
    reactants = reactants.split('.')
    amap2block = {}
    for idx, reac in enumerate(reactants):
        xmol = Chem.MolFromSmiles(reac)
        assert xmol is not None, 'Invalid reactant'
        for x in xmol.GetAtoms():
            amap2block[x.GetAtomMapNum()] = idx

    return amap2block


def get_synthon_smiles(prod, edge_types, mode='break_only'):
    assert mode in ['break_only', 'change'], f'Invalid mode: {mode}'
    if mode == 'break_only':
        break_edges = [k for k, (a, b) in edge_types.items() if b == 0]
        str_synthon = break_fragements(prod, break_edges, canonicalize=False)
        return str_synthon
    else:
        str_synthon = edit_to_synthons(prod, edge_types)
        return str_synthon
