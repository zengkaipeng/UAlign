from rdkit import Chem
from data_utils import load_data
import argparse
from utils.chemistry_parse import clear_map_number
from tqdm import tqdm
from utils.chemistry_parse import (
    get_synthon_edits, get_leaving_group_synthon,
    break_fragements, edit_to_synthons, canonical_smiles,
    eval_by_atom_bond, get_reactants_from_edits
)


def check_n_pos(smi):
    mol = Chem.MolFromSmiles(smi)
    Chem.Kekulize(mol)
    for atom in mol.GetAtoms():
        if atom.GetSymbol() != 'N':
            continue
        bond_vals = sum([x.GetBondTypeAsDouble() for x in atom.GetBonds()])
        if bond_vals == 4 and atom.GetFormalCharge() != 1:
            return False
    return True


def check_charge_show(reac, prod):
    r_mol = Chem.MolFromSmiles(reac)
    p_mol = Chem.MolFromSmiles(prod)

    r_charge = {
        x.GetAtomMapNum(): x.GetFormalCharge()
        for x in r_mol.GetAtoms()
    }
    p_charge = {
        x.GetAtomMapNum(): x.GetFormalCharge()
        for x in p_mol.GetAtoms()
    }

    for k, v in p_charge.items():
        if r_charge[k] != v and v != 0:
            return True

    return False


def get_all_atoms(smi):
    mol = Chem.MolFromSmiles(smi)
    return set(x.GetSymbol() for x in mol.GetAtoms())


MAX_VALENCE = {'N': 3, 'C': 4, 'O': 2, 'Br': 1, 'Cl': 1, 'F': 1, 'I': 1}


def check_rules(smi):
    mol = Chem.MolFromSmiles(smi)
    Chem.Kekulize(mol)
    for atom in mol.GetAtoms():
        this_sym = atom.GetSymbol()
        bond_vals = sum([x.GetBondTypeAsDouble() for x in atom.GetBonds()])
        Hs = atom.GetTotalNumHs()
        this_charge = atom.GetFormalCharge()
        if this_sym == 'C':
            assert bond_vals + Hs - this_charge == MAX_VALENCE['C'], \
                f'Invalid C:{atom.GetAtomMapNum()} of smiles {smi}'

        elif this_sym == 'N':
            assert bond_vals + Hs - this_charge == MAX_VALENCE['N'], \
                f'Invalid N:{atom.GetAtomMapNum()} of smiles {smi}'
        elif this_sym == 'B':
            if len(atom.GetBonds()) == 4 and bond_vals == 4:
                assert this_charge == -1 and atom.GetNumExplicitHs() == 0,\
                    f"Invalid B:{atom.GetAtomMapNum()} of smiles {smi}"

        elif this_sym == 'O':
            assert bond_vals + Hs - this_charge == MAX_VALENCE['O'],\
                f"Invalid O:{atom.GetAtomMapNum()} of smiles {smi}"


def qval_a_mole(reac, prod):
    lgs, syns, conn_edges = get_leaving_group_synthon(
        prod, reac, consider_inner_bonds=True
    )

    modified_atoms, deltsE = get_synthon_edits(
        reac, prod, consider_inner_bonds=True
    )

    # print(f'{reac}>>{prod}')

    syn_str = edit_to_synthons(prod, {k: v[1] for k, v in deltsE.items()})

    if Chem.MolFromSmiles(syn_str) is None:
        print(f'[rxn] {reac}>>{prod}')
        print(f'[edits]', deltsE)
        print(f'[broken reac] {".".join(syns)}')
        print(f'[result] {syn_str}')
        exit()

    syn_str = canonical_smiles(syn_str)
    syns = canonical_smiles('.'.join(syns))

    reactants = get_reactants_from_edits(
        prod_smi=prod, edge_edits={k: v[1] for k, v in deltsE.items()},
        lgs='.'.join(lgs), conns=conn_edges
    )

    if Chem.MolFromSmiles(reactants) is None:
        print(f'[rxn] {reac}>>{prod}')
        print(f'[edits]', deltsE)
        print(f'[broken reac] {".".join(syns)}')
        print(f'[result] {reactants}')
        exit()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('--dir', required=True)
    args = parser.parse_args()

    train_rec, train_prod, train_rxn = load_data(args.dir, 'train')
    val_rec, val_prod, val_rxn = load_data(args.dir, 'val')
    test_rec, test_prod, test_rxn = load_data(args.dir, 'test')

    # n_pos_ood = set()
    # for idx, prod in enumerate(tqdm(train_prod)):
    #     if not check_n_pos(prod):
    #         n_pos_ood.add(clear_map_number(prod))
    #     for x in train_rec[idx].split('.'):
    #         if not check_n_pos(x):
    #             n_pos_ood.add(clear_map_number(x))

    # for idx, prod in enumerate(tqdm(val_prod)):
    #     if not check_n_pos(prod):
    #         n_pos_ood.add(clear_map_number(prod))
    #     for x in val_rec[idx].split('.'):
    #         if not check_n_pos(x):
    #             n_pos_ood.add(clear_map_number(x))

    # for idx, prod in enumerate(tqdm(test_prod)):
    #     if not check_n_pos(prod):
    #         n_pos_ood.add(clear_map_number(prod))
    #     for x in test_rec[idx].split('.'):
    #         if not check_n_pos(x):
    #             n_pos_ood.add(clear_map_number(x))

    # print(n_pos_ood)

    # charge_show_rxn = []
    # for idx, prod in enumerate(tqdm(train_prod)):
    #     if check_charge_show(train_rec[idx], prod):
    #         charge_show_rxn.append(f'{train_rec[idx]}>>{prod}')

    # for idx, prod in enumerate(tqdm(val_prod)):
    #     if check_charge_show(val_rec[idx], prod):
    #         charge_show_rxn.append(f'{val_rec[idx]}>>{prod}')

    # for idx, prod in enumerate(tqdm(test_prod)):
    #     if check_charge_show(test_rec[idx], prod):
    #         charge_show_rxn.append(f'{test_rec[idx]}>>{prod}')

    # all_atoms = set()
    # for idx, prod in enumerate(tqdm(train_prod)):
    #     all_atoms.update(get_all_atoms(train_rec[idx]))
    #     all_atoms.update(get_all_atoms(prod))

    # for idx, prod in enumerate(tqdm(val_prod)):
    #     all_atoms.update(get_all_atoms(val_rec[idx]))
    #     all_atoms.update(get_all_atoms(prod))

    # for idx, prod in enumerate(tqdm(test_prod)):
    #     all_atoms.update(get_all_atoms(test_rec[idx]))
    #     all_atoms.update(get_all_atoms(prod))

    # print(all_atoms)

    for idx, prod in enumerate(tqdm(train_prod)):
        qval_a_mole(train_rec[idx], prod)

    # for idx , prod in enumerate(tqdm(train_prod)):
    #     mol1 = Chem.MolFromSmiles(prod)
    #     for atom in mol1.GetAtoms():
    #         if atom.GetIsAromatic() and atom.GetSymbol() == 'N':
    #             bv = sum(x.GetBondTypeAsDouble() for x in atom.GetBonds())
    #             if bv == 4:
    #                 print(prod, '   ', atom.GetAtomMapNum())
    #     mol2 = Chem.MolFromSmiles(train_rec[idx])
    #     for atom in mol2.GetAtoms():
    #         if atom.GetIsAromatic() and atom.GetSymbol() == 'N':
    #             bv = sum(x.GetBondTypeAsDouble() for x in atom.GetBonds())
    #             if bv == 4:
    #                 print(train_rec[idx], '   ', atom.GetAtomMapNum())
