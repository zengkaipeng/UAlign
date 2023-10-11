import rdkit
from rdkit import Chem

from utils.chemistry_parse import get_modified_atoms_bonds

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

	


