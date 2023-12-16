from rdkit import Chem
from data_utils import load_data
import argparse
from utils.chemistry_parse import clear_map_number
from tqdm import tqdm
from utils.chemistry_parse import (
    get_synthon_edits, get_leaving_group_synthon,
    break_fragements, edit_to_synthons, canonical_smiles,
    eval_by_atom_bond, get_reactants_from_edits,
    clear_map_number, check_aron
)
import json
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers


def get_isomers(smi):
    mol = Chem.MolFromSmiles(smi)
    isomers = tuple(EnumerateStereoisomers(mol))
    isomers_smi = [Chem.MolToSmiles(x, isomericSmiles=True) for x in isomers]
    return isomers_smi


def isomer_match(pred, true):
    true_isomers = get_isomers(true)
    pred_isomers = get_isomers(pred)
    try:
        if set(pred_isomers).issubset(set(true_isomers)) or \
                set(true_isomers).issubset(set(pred_isomers)):
            return True
    except Exception as e:
        pass
    return False


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


def qval_a_mole(reac, prod, all_res):
    lgs, syns, conn_edges = get_leaving_group_synthon(
        prod, reac, consider_inner_bonds=True
    )

    modified_atoms, deltsE = get_synthon_edits(
        reac, prod, consider_inner_bonds=True
    )

    # print(f'{reac}>>{prod}')

    syn_str = edit_to_synthons(prod, {k: v[1] for k, v in deltsE.items()})

    syn_str = canonical_smiles(syn_str)
    syns = canonical_smiles('.'.join(syns))

    # print(f'[rxn] {reac}>>{prod}')
    # print(f'[edits]', deltsE)
    # print(f'[conn]', conn_edges)
    # print(f'[lgs]', '.'.join(lgs))

    reactants = get_reactants_from_edits(
        prod_smi=prod, edge_edits={k: v[1] for k, v in deltsE.items()},
        lgs='.'.join(lgs), conns=conn_edges
    )

    answer1 = clear_map_number(reac)
    answer2 = clear_map_number(reactants)

    answer1_noiso = Chem.MolToSmiles(
        Chem.MolFromSmiles(answer1),
        isomericSmiles=False
    )

    answer2_noiso = Chem.MolToSmiles(
        Chem.MolFromSmiles(answer2),
        isomericSmiles=False
    )

    if answer1_noiso != answer2_noiso:
        all_res.append({
            'rxn': f"{reac}>>{prod}",
            'edit': {f'{a}_{b}': v for (a, b), v in deltsE.items()},
            'syn_str': syn_str,
            'syn_gt': syns,
            'result': answer2,
            'reactants': answer1,
            'lg': '.'.join(lgs)
        })


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
    for idx, prod in enumerate(tqdm(train_prod)):
        if not check_aron(prod):
            print(prod)
            exit()
        if not check_aron(train_rec[idx]):
            print(train_rec[idx])
            exit()

    # for idx, prod in enumerate(tqdm(val_prod)):
    #     all_atoms.update(get_all_atoms(val_rec[idx]))
    #     all_atoms.update(get_all_atoms(prod))

    # for idx, prod in enumerate(tqdm(test_prod)):
    #     all_atoms.update(get_all_atoms(test_rec[idx]))
    #     all_atoms.update(get_all_atoms(prod))

    # print(all_atoms)

    x_all_res = []
    # for idx, prod in enumerate(tqdm(train_prod)):
    #     qval_a_mole(train_rec[idx], prod, x_all_res)

    # print(len(x_all_res))

    # for idx, prod in enumerate(tqdm(val_prod)):
    #     qval_a_mole(val_rec[idx], prod, x_all_res)

    # print(len(x_all_res))

    # for idx, prod in enumerate(tqdm(test_prod)):
    #     qval_a_mole(test_rec[idx], prod, x_all_res)

    # print(len(x_all_res))

    # with open('unmatch.json', 'w') as Fout:
    #     json.dump(x_all_res, Fout, indent=4)

    # print(len(x_all_res))
