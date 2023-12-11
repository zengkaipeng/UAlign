from rdkit import Chem
from data_utils import load_data
import argparse
from utils.chemistry_parse import clear_map_number
from tqdm import tqdm



def check_n_pos(smi):
    mol = Chem.MolFromSmiles(smi)
    for atom in mol.GetAtoms():
        if atom.GetSymbol() != 'N':
            continue
        bond_vals = sum([x.GetBondTypeAsDouble() for x in atom.GetBonds()])
        if bond_vals == 4 and atom.GetFormalCharge() != 1:
            return False
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('--dir', required=True)
    args = parser.parse_args()


    train_rec, train_prod, train_rxn = load_data(args.dir, 'train')
    val_rec, val_prod, val_rxn = load_data(args.dir, 'val')
    test_rec, test_prod, test_rxn = load_data(args.dir, 'test')


    n_pos_ood = set()
    for idx, prod in enumerate(tqdm(train_prod)):
        if not check_n_pos(prod):
            n_pos_ood.add(clear_map_number(prod))
        for x in train_rec[idx].split('.'):
            if not check_n_pos(x):
                n_pos_ood.add(clear_map_number(x))

    for idx, prod in enumerate(tqdm(val_prod)):
        if not check_n_pos(prod):
            n_pos_ood.add(clear_map_number(prod))
        for x in val_rec[idx].split('.'):
            if not check_n_pos(x):
                n_pos_ood.add(clear_map_number(x))
    
    for idx, prod in enumerate(tqdm(test_prod)):
        if not check_n_pos(prod):
            n_pos_ood.add(clear_map_number(prod))
        for x in test_rec[idx].split('.'):
            if not check_n_pos(x):
                n_pos_ood.add(clear_map_number(x))

    print(len(n_pos_ood))
    