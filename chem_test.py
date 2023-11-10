import rdkit
from rdkit import Chem

from utils.chemistry_parse import get_modified_atoms_bonds
from draws import rxn2svg

if __name__ == '__main__':
	mol = 'c1ccc2ccccc2c1'
	mol = Chem.MolFromSmiles(mol)
	mol2 = 'C1=CCc2ccccc2C1'
	mol2 = Chem.MolFromSmiles(mol2)
	# print(Chem.MolToSmiles(mol2))
	
	rxn = '[CH:1]1=[CH:2][CH2:3][c:4]2[cH:5][cH:6][cH:7][cH:8][c:9]2[CH2:10]1>>[cH:1]1[cH:2][cH:3][c:4]2[cH:5][cH:6][cH:7][cH:8][c:9]2[cH:10]1'

	reac, prod = rxn.split('>>')
	print('[INFO] not keku')
	print(get_modified_atoms_bonds(reac, prod, kekulize=False))

	print('[INFO] kekulize')
	print(get_modified_atoms_bonds(reac, prod, kekulize=True))

	rxn = '[O:13]=[CH:10][CH2:11][CH3:12].[O:14]=[CH:1][c:2]1[c:3]([NH2:4])[cH:5][c:6]([Cl:7])[cH:8][cH:9]1>>[cH:1]1[c:2]2[c:3]([n:4][cH:10][c:11]1[CH3:12])[cH:5][c:6]([Cl:7])[cH:8][cH:9]2'
	reac, prod = rxn.split('>>')
	print(get_modified_atoms_bonds(reac, prod, kekulize=True))
	rxn2svg(rxn, 'tmp_figs/aromatic.svg')


