import rdkit
from rdkit import Chem
from utils.chemistry_parse import clear_map_number
from utils.chemistry_parse import convert_res_into_smiles
from draws import mol2svg


omol = '[CH3:1][C:2]1([c:3]2[cH:4][cH:5][cH:6][o:7]2)[O:8][CH2:9][CH2:10][O:11]1'
mol = Chem.MolFromSmiles(omol)

tmol = Chem.RWMol(mol)

for bond in mol.GetBonds():
    a_start = bond.GetBeginAtom().GetAtomMapNum()
    a_end = bond.GetEndAtom().GetAtomMapNum()
    beg_idx = bond.GetBeginAtom().GetIdx()
    end_idx = bond.GetEndAtom().GetIdx()

    if (a_start, a_end) in [(2, 11), (11, 2)]:
        tmol.RemoveBond(beg_idx, end_idx)
    if (a_start, a_end) in [(8, 9), (9, 8)]:
        tmol.RemoveBond(beg_idx, end_idx)

    if (a_start, a_end) in [(2, 8), (8, 2)]:
        tmol.RemoveBond(beg_idx, end_idx)
        tmol.AddBond(beg_idx, end_idx, Chem.rdchem.BondType.DOUBLE)


mol = tmol.GetMol()

print(Chem.MolToSmiles(mol))

print(clear_map_number(Chem.MolToSmiles(mol)))


mol2svg(Chem.MolToSmiles(mol), output_path='tmp_figs/synthon1.svg')


print(convert_res_into_smiles({
    1: 18, 2: 18, 3: 18, 4: 18, 5: 18, 6: 18,
    7: 12, 8: 12
}, {
    (1, 2): 1, (2, 3): 1, (2, 8): 2, (3, 7): 1,
    (4, 3): 2, (7, 6): 1, (5, 6): 2, (4, 5): 1
}, {}, {}
))



print(clear_map_number(omol))


xmol = '[CH2]C[O]'
xmol = Chem.MolFromSmiles(xmol)
print(Chem.MolToSmiles(xmol))
