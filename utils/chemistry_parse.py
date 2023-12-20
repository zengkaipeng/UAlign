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

MAX_VALENCE = {'N': 3, 'C': 4, 'O': 2, 'Br': 1, 'Cl': 1, 'F': 1, 'I': 1}


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


def break_fragements(smiles, break_edges):
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

    amap = {x.GetAtomMapNum(): x.GetIdx() for x in Mol.GetAtoms()}
    for a, b in break_edges:
        start_idx, end_idx = amap[a], amap[b]
        if tmol.GetBondBetweenAtoms(start_idx, end_idx) is not None:
            tmol.RemoveBond(start_idx, end_idx)

    answer = Chem.MolToSmiles(tmol.GetMol())
    if Chem.MolFromSmiles(answer) is None:
        print('\n[smi]', smiles)
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
                    (start_amap, end_amap) not in prod_bonds:
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




def edit_to_synthons(smi, edge_edits):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        raise ValueError(f'Invalid smiles {smi} given')

    assert all(x.GetAtomMapNum() != 0 for x in mol.GetAtoms()), \
        'atom mapping are required for editing the mol'

    old_types = {}
    for bond in mol.GetBonds():
        a_atom, b_atom = bond.GetBeginAtom(), bond.GetEndAtom()
        a_amap, b_amap = a_atom.GetAtomMapNum(), b_atom.GetAtomMapNum()
        old_types[(a_amap, b_amap)] = old_types[(b_amap, a_amap)] = \
            bond.GetBondTypeAsDouble()

    old_ExpHs = {
        x.GetAtomMapNum(): x.GetNumExplicitHs() for x in mol.GetAtoms()
    }
    Chem.Kekulize(mol, True)
    # clear all the aromatic information, to avoid invalid breaking

    amap = {x.GetAtomMapNum(): x.GetIdx() for x in mol.GetAtoms()}
    ke_old_bond_vals = {
        x.GetAtomMapNum(): sum(y.GetBondTypeAsDouble() for y in x.GetBonds())
        for x in mol.GetAtoms()
    }

    tmol = Chem.RWMol(mol)

    modified_atoms = set()

    for (a, b), n_type in edge_edits.items():
        if n_type == old_types.get((a, b), 0):
            continue
        a_idx, b_idx = amap[a], amap[b]
        begin_atom = tmol.GetAtomWithIdx(a_idx)
        end_atom = tmol.GetAtomWithIdx(b_idx)
        a_sym, b_sym = begin_atom.GetSymbol(), end_atom.GetSymbol()
        old_bond = tmol.GetBondBetweenAtoms(a_idx, b_idx)
        if old_bond is not None:
            tmol.RemoveBond(a_idx, b_idx)
            modified_atoms.update((a, b))

            if old_types[(a, b)] == 1 and a_sym == 'N' and b_sym == 'O':
                if begin_atom.GetFormalCharge() == 1:
                    begin_atom.SetFormalCharge(0)
                    begin_atom.SetNumExplicitHs(0)
                if end_atom.GetFormalCharge() == -1:
                    end_atom.SetFormalCharge(0)
            elif old_types[(a, b)] == 1 and a_sym == 'O' and b_sym == 'N':
                if begin_atom.GetFormalCharge() == -1:
                    begin_atom.SetFormalCharge(0)
                if end_atom.GetFormalCharge() == 1:
                    end_atom.SetFormalCharge(0)
                    end_atom.SetNumExplicitHs(0)

    broken_smi = Chem.MolToSmiles(tmol.GetMol())
    broken_mol = Chem.MolFromSmiles(broken_smi)
    broken_amap = {
        x.GetAtomMapNum(): x.GetIdx() for x in broken_mol.GetAtoms()
    }

    tmol = Chem.RWMol(broken_mol)

    for atom in tmol.GetAtoms():
        ax = atom.GetAtomMapNum()
        atom.SetNumExplicitHs(old_ExpHs[ax])

    tmol.UpdatePropertyCache()

    # reload the aromatic edges
    # recover the num of explicit Hs, solve the conflict of exp atoms

    for (a, b), n_type in edge_edits.items():
        if n_type == old_types.get((a, b), 0) or n_type == 0:
            continue
        assert n_type != 1.5, "Building aromatic bonds"
        a_idx, b_idx = broken_amap[a], broken_amap[b]
        begin_atom = tmol.GetAtomWithIdx(a_idx)
        end_atom = tmol.GetAtomWithIdx(b_idx)
        a_sym, b_sym = begin_atom.GetSymbol(), end_atom.GetSymbol()

        tmol.AddBond(a_idx, b_idx, BOND_FLOAT_TO_TYPE[n_type])
        modified_atoms.update((a, b))

        if old_types.get((a, b), 0) == 1 and n_type == 2 and a_sym == 'S'\
                and b_sym == 'O' and end_atom.GetFormalCharge() == -1:
            end_atom.SetFormalCharge(0)
        elif old_types.get((a, b), 0) == 1 and n_type == 2 and a_sym == 'O'\
                and b_sym == 'S' and begin_atom.GetFormalCharge() == -1:
            begin_atom.SetFormalCharge(-1)

    for ax in modified_atoms:
        atom = tmol.GetAtomWithIdx(broken_amap[ax])
        curr_bv = sum(y.GetBondTypeAsDouble() for y in atom.GetBonds())
        if curr_bv >= ke_old_bond_vals[ax]:
            delta = curr_bv - ke_old_bond_vals[ax]
            curr_hs = int(max(0, atom.GetNumExplicitHs() - delta))
            atom.SetNumExplicitHs(curr_hs)

    for atom in tmol.GetAtoms():
        if atom.GetSymbol() == 'N':
            bond_vals = sum([x.GetBondTypeAsDouble() for x in atom.GetBonds()])
            if bond_vals == 4 and not atom.GetIsAromatic():
                atom.SetFormalCharge(1)
        elif atom.GetSymbol() == 'P':
            bond_vals = [x.GetBondTypeAsDouble() for x in atom.GetBonds()]
            if sum(bond_vals) == 4 and len(bond_vals) == 4:
                atom.SetFormalCharge(1)
                atom.SetNumExplicitHs(0)

    syn_mol = tmol.GetMol()
    answer = Chem.MolToSmiles(syn_mol)
    return answer

    # updated_bond_types = {
    #     (a, b): c for (a, b), c in old_types.items() if a < b
    # }
    # updated_bond_types.update(edge_edits)

    # return rebuild_aromatic(answer, updated_bond_types)


def get_reactants_from_edits(prod_smi, edge_edits, lgs, conns):
    prod_mol = Chem.MolFromSmiles(prod_smi)
    lg_mol = Chem.MolFromSmiles(lgs)
    if lg_mol is None or prod_mol is None:
        raise ValueError(f'Invalid Smiles passed, {prod_smi}\n{lgs}')

    assert all(x.GetAtomMapNum() != 0 for x in prod_mol.GetAtoms()), \
        'atom mapping are required for editing the mol'

    assert all(x.GetAtomMapNum() != 0 for x in lg_mol.GetAtoms()), \
        'atom mapping are required for editing the mol'

    max_amap = max(x.GetAtomMapNum() for x in prod_mol.GetAtoms())
    delt_amap = max_amap + 100

    real_conns = {(x, y + delt_amap): v for (x, y), v in conns.items()}

    for atom in lg_mol.GetAtoms():
        curr_anum = atom.GetAtomMapNum()
        atom.ClearProp('molAtomMapNumber')
        atom.SetAtomMapNum(curr_anum + delt_amap)

    new_mol = Chem.Mol(prod_mol)
    new_mol = Chem.CombineMols(new_mol, lg_mol)

    old_types = {}
    for bond in new_mol.GetBonds():
        a_atom, b_atom = bond.GetBeginAtom(), bond.GetEndAtom()
        a_amap, b_amap = a_atom.GetAtomMapNum(), b_atom.GetAtomMapNum()
        old_types[(a_amap, b_amap)] = old_types[(b_amap, a_amap)] = \
            bond.GetBondTypeAsDouble()

    old_ExpHs = {
        x.GetAtomMapNum(): x.GetNumExplicitHs() for x in new_mol.GetAtoms()
    }
    Chem.Kekulize(new_mol, True)
    # clear all the aromatic information, to avoid invalid breaking

    amap = {x.GetAtomMapNum(): x.GetIdx() for x in new_mol.GetAtoms()}
    ke_old_bond_vals = {
        x.GetAtomMapNum(): sum(y.GetBondTypeAsDouble() for y in x.GetBonds())
        for x in new_mol.GetAtoms()
    }

    tmol = Chem.RWMol(new_mol)

    modified_atoms = set()

    for (a, b), n_type in edge_edits.items():
        if n_type == old_types.get((a, b), 0):
            continue
        a_idx, b_idx = amap[a], amap[b]
        begin_atom = tmol.GetAtomWithIdx(a_idx)
        end_atom = tmol.GetAtomWithIdx(b_idx)
        a_sym, b_sym = begin_atom.GetSymbol(), end_atom.GetSymbol()
        old_bond = tmol.GetBondBetweenAtoms(a_idx, b_idx)
        if old_bond is not None:
            tmol.RemoveBond(a_idx, b_idx)
            modified_atoms.update((a, b))

            if old_types[(a, b)] == 1 and a_sym == 'N' and b_sym == 'O':
                if begin_atom.GetFormalCharge() == 1:
                    begin_atom.SetFormalCharge(0)
                    begin_atom.SetNumExplicitHs(0)
                    ke_old_bond_vals[a] -= 1
                if end_atom.GetFormalCharge() == -1:
                    end_atom.SetFormalCharge(0)
                    ke_old_bond_vals[b] += 1
            elif old_types[(a, b)] == 1 and a_sym == 'O' and b_sym == 'N':
                if begin_atom.GetFormalCharge() == -1:
                    begin_atom.SetFormalCharge(0)
                    ke_old_bond_vals[a] += 1
                if end_atom.GetFormalCharge() == 1:
                    end_atom.SetFormalCharge(0)
                    end_atom.SetNumExplicitHs(0)
                    ke_old_bond_vals[b] -= 1

    broken_smi = Chem.MolToSmiles(tmol.GetMol())
    broken_mol = Chem.MolFromSmiles(broken_smi)
    broken_amap = {
        x.GetAtomMapNum(): x.GetIdx() for x in broken_mol.GetAtoms()
    }

    tmol = Chem.RWMol(broken_mol)

    for atom in tmol.GetAtoms():
        ax = atom.GetAtomMapNum()
        atom.SetNumExplicitHs(old_ExpHs[ax])

    tmol.UpdatePropertyCache()

    # reload the aromatic edges
    # recover the num of explicit Hs, solve the conflict of exp atoms

    for (a, b), n_type in edge_edits.items():
        if n_type == old_types.get((a, b), 0) or n_type == 0:
            continue
        assert n_type != 1.5, "Building aromatic bonds"
        a_idx, b_idx = broken_amap[a], broken_amap[b]
        begin_atom = tmol.GetAtomWithIdx(a_idx)
        end_atom = tmol.GetAtomWithIdx(b_idx)
        a_sym, b_sym = begin_atom.GetSymbol(), end_atom.GetSymbol()

        tmol.AddBond(a_idx, b_idx, BOND_FLOAT_TO_TYPE[n_type])
        modified_atoms.update((a, b))

        if old_types.get((a, b), 0) == 1 and n_type == 2 and a_sym == 'S'\
                and b_sym == 'O' and end_atom.GetFormalCharge() == -1:
            end_atom.SetFormalCharge(0)
        elif old_types.get((a, b), 0) == 1 and n_type == 2 and a_sym == 'O'\
                and b_sym == 'S' and begin_atom.GetFormalCharge() == -1:
            begin_atom.SetFormalCharge(-1)

    for (a, b), v in real_conns.items():
        assert v != 1.5, "Building aromatic bonds"
        a_idx, b_idx = broken_amap[a], broken_amap[b]
        begin_atom = tmol.GetAtomWithIdx(a_idx)
        end_atom = tmol.GetAtomWithIdx(b_idx)

        tmol.AddBond(a_idx, b_idx, BOND_FLOAT_TO_TYPE[v])
        modified_atoms.add(a)

    for ax in modified_atoms:
        atom = tmol.GetAtomWithIdx(broken_amap[ax])
        curr_bv = sum(y.GetBondTypeAsDouble() for y in atom.GetBonds())
        if curr_bv >= ke_old_bond_vals[ax]:
            delta = curr_bv - ke_old_bond_vals[ax]
            curr_hs = int(max(0, atom.GetNumExplicitHs() - delta))
            atom.SetNumExplicitHs(curr_hs)
        else:
            delta = ke_old_bond_vals[ax] - curr_bv
            curr_hs = int(atom.GetNumExplicitHs() + delta)
            atom.SetNumExplicitHs(curr_hs)

    for atom in tmol.GetAtoms():
        if atom.GetSymbol() == 'C':
            bond_vals = sum([x.GetBondTypeAsDouble() for x in atom.GetBonds()])
            nei_sym = [nei.GetSymbol() for nei in atom.GetNeighbors()]
            check1 = 'O' in nei_sym and 'N' in nei_sym and \
                atom.GetIsAromatic() and len(atom.GetBonds()) == 3

            check2 = 'O' in nei_sym and len(atom.GetBonds()) == 3 \
                and atom.GetIsAromatic()

            check3 = 'N' in nei_sym and len(atom.GetBonds()) == 3 \
                and atom.GetIsAromatic()
            check4 = 'S' in nei_sym and len(atom.GetBonds()) == 3 \
                and atom.GetIsAromatic()

            if check1 or check2 or check3 or check4:
                if bond_vals >= MAX_VALENCE['C']:
                    atom.SetNumExplicitHs(0)
            else:
                if bond_vals >= MAX_VALENCE['C']:
                    atom.SetNumExplicitHs(0)
                    atom.SetFormalCharge(int(bond_vals - MAX_VALENCE['C']))
                else:
                    atom.SetNumExplicitHs(int(MAX_VALENCE['C'] - bond_vals))
                    atom.SetFormalCharge(0)

        elif atom.GetSymbol() == 'N':
            bond_vals = sum([x.GetBondTypeAsDouble() for x in atom.GetBonds()])
            if not atom.GetIsAromatic():
                if bond_vals >= MAX_VALENCE['N']:
                    atom.SetNumExplicitHs(0)
                    atom.SetFormalCharge(int(bond_vals - MAX_VALENCE['N']))
            elif atom.GetFormalCharge() == 1:
                if bond_vals == MAX_VALENCE['N']:
                    atom.SetFormalCharge(0)
                    atom.SetNumExplicitHs(0)

        elif atom.GetSymbol() == 'S':
            bond_vals = sum([x.GetBondTypeAsDouble() for x in atom.GetBonds()])
            if not atom.GetIsAromatic() and bond_vals in [2, 4, 6]:
                atom.SetFormalCharge(0)
                atom.SetNumExplicitHs(0)
            elif atom.GetIsAromatic() and bond_vals in [3, 5]:
                atom.SetNumExplicitHs(0)

        elif atom.GetSymbol() == 'Sn':
            bond_vals = sum([x.GetBondTypeAsDouble() for x in atom.GetBonds()])
            if bond_vals >= 4:
                atom.SetNumExplicitHs(0)
        elif atom.GetSymbol() == 'O':
            bond_vals = sum([x.GetBondTypeAsDouble() for x in atom.GetBonds()])
            if bond_vals >= MAX_VALENCE['O']:
                atom.SetNumExplicitHs(0)
            elif bond_vals == 0:
                atom.SetNumExplicitHs(2)
            elif bond_vals < MAX_VALENCE['O'] and atom.GetFormalCharge() != -1:
                atom.SetNumExplicitHs(int(MAX_VALENCE['O'] - bond_vals))
                atom.SetFormalCharge(0)
        elif atom.GetSymbol() == 'B':
            bond_vals = [x.GetBondTypeAsDouble() for x in atom.GetBonds()]
            if len(bond_vals) == 4 and sum(bond_vals) == 4:
                atom.SetFormalCharge(-1)
                atom.SetNumExplicitHs(0)
            elif sum(bond_vals) >= 3:
                atom.SetNumExplicitHs(0)
        elif atom.GetSymbol() in ['Br', 'Cl', 'I', 'F']:
            bond_vals = sum([x.GetBondTypeAsDouble() for x in atom.GetBonds()])
            if bond_vals >= MAX_VALENCE[atom.GetSymbol()]:
                atom.SetNumExplicitHs(0)

    reac_mol = tmol.GetMol()
    return Chem.MolToSmiles(reac_mol)


def run_special_case(reactants):
    azide_rule = AllChem.ReactionFromSmarts(
        '[NH:2]=[N+:3]=[N-:4]>>[NH0-:2]=[N+:3]=[N-:4]')
    reac_mols = [Chem.MolFromSmiles(x) for x in reactants.split('.')]
    for idx, mol in enumerate(reac_mols):
        out = azide_rule.RunReactants((mol, ))
        if len(out) > 0:
            reac_mols[idx] = out[0][0]

    px = ".".join([Chem.MolToSmiles(x) for x in reac_mols])
    return canonical_smiles(px)


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


def eval_by_atom_bond(smi1, smi2):
    mol1, mol2 = get_mol(smi1), get_mol(smi2)
    ke_mol1 = get_mol(smi1, kekulize=True)
    ke_mol2 = get_mol(smi2, kekulize=True)

    bond1 = get_bond_info(mol1)
    bond2 = get_bond_info(mol2)
    ke_bond1 = get_bond_info(ke_mol1)
    ke_bond2 = get_bond_info(ke_mol2)

    if set(bond1.keys()) != set(bond2.keys()):
        return False

    for k, v in bond1.items():
        if v[0] != bond2[k][0] and ke_bond1[k][0] != ke_bond2[k][0]:
            # print('die', k, ke_bond1[k][0], ke_bond2[k][0])
            return False

    return True
