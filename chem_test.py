import rdkit
from rdkit import Chem

from utils.chemistry_parse import get_modified_atoms_bonds, get_synthons
from draws import rxn2svg
from data_utils import load_data


if __name__ == '__main__':
    # mol = 'c1ccc2ccccc2c1'
    # mol = Chem.MolFromSmiles(mol)
    # mol2 = 'C1=CCc2ccccc2C1'
    # mol2 = Chem.MolFromSmiles(mol2)
    # # print(Chem.MolToSmiles(mol2))

    # rxn = '[CH:1]1=[CH:2][CH2:3][c:4]2[cH:5][cH:6][cH:7][cH:8][c:9]2[CH2:10]1>>[cH:1]1[cH:2][cH:3][c:4]2[cH:5][cH:6][cH:7][cH:8][c:9]2[cH:10]1'

    # reac, prod = rxn.split('>>')
    # print('[INFO] not keku')
    # print(get_modified_atoms_bonds(reac, prod, kekulize=False))

    # print('[INFO] kekulize')
    # print(get_modified_atoms_bonds(reac, prod, kekulize=True))

    # rxn = '[O:13]=[CH:10][CH2:11][CH3:12].[O:14]=[CH:1][c:2]1[c:3]([NH2:4])[cH:5][c:6]([Cl:7])[cH:8][cH:9]1>>[cH:1]1[c:2]2[c:3]([n:4][cH:10][c:11]1[CH3:12])[cH:5][c:6]([Cl:7])[cH:8][cH:9]2'
    # reac, prod = rxn.split('>>')
    # print(get_modified_atoms_bonds(reac, prod, kekulize=True))
    # rxn2svg(rxn, 'tmp_figs/aromatic.svg')

    # train_rec, train_prod, train_rxn = load_data('../data/UTPSO-50K', 'train')
    # val_rec, val_prod, val_rxn = load_data('../data/UTPSO-50K', 'val')
    # test_rec, test_prod, test_rxn = load_data('../data/UTPSO-50K', 'test')

    # all_deltH = set()
    # for idx, prod in enumerate(train_prod):
    #     deltH, _ = get_synthons(prod, train_rec[idx])
    #     all_deltH.update(deltH.values())

    # for idx, prod in enumerate(val_prod):
    #     deltH, _ = get_synthons(prod, val_rec[idx])
    #     all_deltH.update(deltH.values())


    # for idx, prod in enumerate(test_prod):
    #     deltH, _ = get_synthons(prod, test_rec[idx])
    #     all_deltH.update(deltH.values())

    # print(all_deltH)
    
    tmol = Chem.RWMol()
    last_idx, fst_idx = None, None
    for i in range(6):
    	atom = Chem.Atom("C")
    	atom.SetAtomMapNum(i + 1)
    	curr_idx = tmol.AddAtom(atom)

    for i in range(6):
    	dst = i + 1 if i != 5 else 0
    	if i & 1:
    		tmol.AddBond(i, dst, Chem.rdchem.BondType.DOUBLE)
    	else:
    		tmol.AddBond(i, dst, Chem.rdchem.BondType.SINGLE)

    mol = tmol.GetMol()
    t_str = Chem.MolToSmiles(mol)
    mol = Chem.MolFromSmiles(t_str)
    print(Chem.MolToSmiles(mol))




