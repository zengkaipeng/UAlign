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
# ATOM_TPYE_TO_IDX = {
#     'S_1_SP3': 1, 'O_0_SP3': 2, 'S_0_SP3D2': 3, 'Cu_0_SP3D2': 4, 'N_1_SP3': 5,
#     'S_0_SP3D': 6, 'Br_0_SP3': 7, 'C_0_SP': 8, 'N_0_SP2': 9, 'S_0_SP3': 10,
#     'P_0_SP3': 11, 'Sn_0_SP3': 12, 'I_0_SP3': 13, 'S_0_SP2': 14, 'C_0_SP3': 15,
#     'P_0_SP2': 16, 'C_0_SP2': 17, 'S_-1_SP2': 18, 'C_-1_SP': 19, 'F_0_SP3': 20,
#     'O_-1_SP3': 21, 'Mg_1_S': 22, 'Mg_0_SP': 23, 'N_-1_SP2': 24, 'O_-1_SP2': 25,
#     'S_1_SP2': 26, 'Zn_0_SP': 27, 'Se_0_SP2': 28, 'Zn_1_S': 29, 'Si_0_SP3': 30,
#     'N_0_SP': 31, 'N_1_SP2': 32, 'P_1_SP3': 33, 'P_0_SP3D': 34, 'O_0_SP2': 35,
#     'N_1_SP': 36, 'S_-1_SP3': 37, 'Se_0_SP3': 38, 'Cl_0_SP3': 39, 'P_1_SP2': 40,
#     'B_0_SP2': 41, 'N_0_SP3': 42
# }

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


def get_leaving_group(prod: str, reac: str):
    prod_amap = get_all_amap(prod)
    reac_amap = get_all_amap(reac)

    r_mol = get_mol(reac)
    if r_mol is None:
        return [], []
    break_edges, conn_egs = [], []
    for bond in r_mol.GetBonds():
        start_atom = bond.GetBeginAtom()
        end_atom = bond.GetEndAtom()
        start_amap = start_atom.GetAtomMapNum()
        end_amap = end_atom.GetAtomMapNum()

        if start_amap in prod_amap and end_amap not in prod_amap:
            break_edges.append((start_amap, end_amap))
            break_edges.append((end_amap, start_amap))
            conn_egs.append((start_amap, end_amap))
        if start_amap not in prod_amap and end_amap in prod_amap:
            break_edges.append((start_amap, end_amap))
            break_edges.append((end_amap, start_amap))
            conn_egs.append((end_amap, start_amap))

    frgs = break_fragements(reac, break_edges).split('.')
    answer = []
    for lg in frgs:
        all_amap = get_all_amap(lg)
        if len(all_amap & prod_amap) == 0:
            answer.append(lg)
        else:
            assert len(all_amap & prod_amap) == len(all_amap), \
                f'The breaking is not correct, {reac}>>{prod}'
    return answer, conn_egs


def get_synthons(prod: str, reac: str, kekulize: bool = False):
    reac_mol, prod_mol = get_mol(reac), get_mol(prod)
    if reac_mol is None or prod_mol is None:
        return {}, {}
    if kekulize:
        reac_mol, prod_mol = align_kekule_pairs(reac, prod)
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

    atom2deltaH, edges2typechange = {}, {}
    for atom in prod_mol.GetAtoms():
        amap_num = atom.GetAtomMapNum()
        numHs_prod = atom.GetTotalNumHs()
        reac_atom = reac_mol.GetAtomWithIdx(reac_amap_idx[amap_num])
        numHs_reac = reac_atom.GetTotalNumHs()
        atom2deltaH[amap_num] = numHs_prod - numHs_reac

    for bond in prod_bonds:
        target_type = reac_bonds[bond][0] if bond in reac_bonds else 0.0
        edges2typechange[bond] = (prod_bonds[bond][0], target_type)

    # We have omitted the formation of new bonds between the product atoms
    # during the reaction process, as this situation occurs
    # only to a small extent.

    return atom2deltaH, edges2typechange


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


def extend_by_dfs(reac, activate_nodes, prod_amap):
    def dfs(mol, x, vis, curr_nodes):
        curr = mol.GetAtomWithIdx(x)
        curr_amap = curr.GetAtomMapNum()
        if curr_amap in vis:
            return
        curr_nodes.append(curr_amap)
        vis.add(curr_amap)
        for nei in curr.GetNeighbors():
            dfs(mol, nei.GetIdx(), vis, curr_nodes)

    curr_nodes = [-1 for _ in range(max(prod_amap.values()) + 1)]
    for k, v in prod_amap.items():
        curr_nodes[v] = k

    vis = set(prod_amap.keys())
    assert all(x != -1 for x in curr_nodes), 'Invalid prod_amap'

    mol = Chem.MolFromSmiles(reac)
    if mol is None:
        raise ValueError(f'Invalid smiles {reac}')

    # mark connected parts
    for atom in mol.GetAtoms():
        am = atom.GetAtomMapNum()
        if am in activate_nodes:
            for nei in atom.GetNeighbors():
                dfs(mol, nei.GetIdx(), vis, curr_nodes)

    # mark isolated part
    for atom in mol.GetAtoms():
        am = atom.GetAtomMapNum()
        if am not in vis:
            dfs(mol, atom.GetIdx(), vis, curr_nodes)

    return {v: idx for idx, v in enumerate(curr_nodes)}


def extend_by_bfs(reac, activate_nodes, prod_amap):

    def bfs_with_Q(Q, lf, mol, vis):
        while lf < len(Q):
            top = Q[lf]
            top_atom = mol.GetAtomWithIdx(top)
            # print('[top atom]', lf, top_atom.GetAtomMapNum())
            for nei in top_atom.GetNeighbors():
                nei_amap = nei.GetAtomMapNum()
                if nei_amap not in vis:
                    vis.add(nei_amap)
                    curr_nodes.append(nei_amap)
                    Q.append(nei.GetIdx())
            lf += 1

        # print('\n[done]\n')

    curr_nodes = [-1 for _ in range(max(prod_amap.values()) + 1)]
    for k, v in prod_amap.items():
        curr_nodes[v] = k
    vis = set(prod_amap.keys())
    # print(vis)

    assert all(x != -1 for x in curr_nodes), 'Invalid prod_amap'

    mol = Chem.MolFromSmiles(reac)
    if mol is None:
        raise ValueError(f'Invalid smiles {reac}')

    # mark connected part
    Q, lf = [], 0

    for atom in mol.GetAtoms():
        am = atom.GetAtomMapNum()
        if am in activate_nodes:
            Q.append(atom.GetIdx())

    bfs_with_Q(Q, lf, mol, vis)

    # mark isolated part

    for atom in mol.GetAtoms():
        am = atom.GetAtomMapNum()
        if am not in vis:
            Q = [atom.GetIdx()]
            vis.add(am)
            curr_nodes.append(am)
            bfs_with_Q(Q, 0, mol, vis)

    return {v: idx for idx, v in enumerate(curr_nodes)}


def add_random_Amap(smiles):
    mol = Chem.MolFromSmiles(smiles)
    for x in mol.GetAtoms():
        x.ClearProp('molAtomMapNumber')
    for idx, x in enumerate(mol.GetAtoms()):
        x.SetAtomMapNum(idx + 1)
    return Chem.MolToSmiles(mol)


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
