import rdkit
from rdkit import Chem
from utils.chemistry_parse import clear_map_number, canonical_smiles
from utils.chemistry_parse import convert_res_into_smiles
from draws import mol2svg
from utils.chemistry_parse import add_random_Amap


omol = '[CH3:1][C:2]1([c:3]2[cH:4][cH:5][cH:6][o:7]2)[O:8][CH2:9][CH2:10][O:11]1'
mol = Chem.MolFromSmiles(omol)
tmol = Chem.RWMol(mol)

for bond in mol.GetBonds():
    a_start = bond.GetBeginAtom().GetAtomMapNum()
    a_end = bond.GetEndAtom().GetAtomMapNum()
    beg_idx = bond.GetBeginAtom().GetIdx()
    end_idx = bond.GetEndAtom().GetIdx()

    
    if (a_start, a_end) in [(2, 11), (11, 2)]:
        print(beg_idx, end_idx)

        tmol.RemoveBond(beg_idx, end_idx)
    if (a_start, a_end) in [(8, 9), (9, 8)]:
        print(beg_idx, end_idx)

        tmol.RemoveBond(beg_idx, end_idx)
    if (a_start, a_end) in [(2, 8), (8, 2)]:
        print(beg_idx, end_idx)

        tmol.RemoveBond(beg_idx, end_idx)
        tmol.AddBond(beg_idx, end_idx, Chem.rdchem.BondType.DOUBLE)



print([Chem.MolToSmiles(x) for x in Chem.GetMolFrags(tmol, asMols=True)])


mol = tmol.GetMol()

print(Chem.MolToSmiles(mol))

print(clear_map_number(Chem.MolToSmiles(mol)))


mol2svg(Chem.MolToSmiles(mol), output_path='tmp_figs/synthon1.svg')


# print(convert_res_into_smiles({
#     1: 18, 2: 18, 3: 18, 4: 18, 5: 18, 6: 18,
#     7: 12, 8: 12
# }, {
#     (1, 2): 1, (2, 3): 1, (2, 8): 2, (3, 7): 1,
#     (4, 3): 2, (7, 6): 1, (5, 6): 2, (4, 5): 1
# }, {}, {}
# ))


std_mol_s = clear_map_number(omol)

print('std', std_mol_s)
mol2svg(std_mol_s, output_path='tmp_figs/org.svg')
std_mol = Chem.MolFromSmiles(std_mol_s)


std_mol = Chem.RemoveHs(std_mol)
tmol = Chem.RWMol(std_mol)

for bond in std_mol.GetBonds():
    beg_idx = bond.GetBeginAtom().GetIdx()
    end_idx = bond.GetEndAtom().GetIdx()
    print(beg_idx, end_idx, bond.GetIdx())

    if (beg_idx, end_idx) == (10, 1):
        tmol.RemoveBond(beg_idx, end_idx)
    if (beg_idx, end_idx) == (7, 8):
        tmol.RemoveBond(beg_idx, end_idx)
    if (beg_idx, end_idx) == (1, 7):
        tmol.RemoveBond(beg_idx, end_idx)
        tmol.AddBond(beg_idx, end_idx, Chem.rdchem.BondType.DOUBLE)


print([Chem.MolToSmiles(x) for x in Chem.GetMolFrags(tmol, asMols=True)])


# frgs = Chem.FragmentOnBonds(std_mol, (10, 6, 7))
# print(Chem.MolToSmiles(frgs))


mol = tmol.GetMol()
print('final', canonical_smiles(Chem.MolToSmiles(mol)))




std_mol_r = add_random_Amap(std_mol_s)
std_mol = Chem.MolFromSmiles(std_mol_r)
tmol = Chem.RWMol(std_mol)

print(std_mol_r)

for bond in std_mol.GetBonds():
    beg_idx = bond.GetBeginAtom().GetIdx()
    end_idx = bond.GetEndAtom().GetIdx()
    print(beg_idx, end_idx, bond.GetIdx())

    if (beg_idx, end_idx) == (10, 1):
        tmol.RemoveBond(beg_idx, end_idx)
    if (beg_idx, end_idx) == (7, 8):
        tmol.RemoveBond(beg_idx, end_idx)
    if (beg_idx, end_idx) == (1, 7):
        tmol.RemoveBond(beg_idx, end_idx)
        tmol.AddBond(beg_idx, end_idx, Chem.rdchem.BondType.DOUBLE)


print([Chem.MolToSmiles(x) for x in Chem.GetMolFrags(tmol, asMols=True)])

print(clear_map_number(Chem.MolToSmiles(tmol.GetMol())))


# for atom in Chem.MolFromSmiles(omol).GetAtoms():
#     idx = atom.GetIdx()
#     print(idx, atom.GetNumExplicitHs(), atom.GetNumImplicitHs(), atom.GetTotalNumHs())

# print()
# for atom in Chem.MolFromSmiles(std_mol_s).GetAtoms():
#     idx = atom.GetIdx()
#     print(idx, atom.GetNumExplicitHs(), atom.GetNumImplicitHs(), atom.GetTotalNumHs())
# print()

# for atom in mol.GetAtoms():
#     idx = atom.GetIdx()
#     print(idx, atom.GetNumExplicitHs(), atom.GetNumImplicitHs(), atom.GetTotalNumHs())
# print()


# r_mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
# for atom in r_mol.GetAtoms():
#     idx = atom.GetIdx()
#     print(idx, atom.GetNumExplicitHs(), atom.GetNumImplicitHs(), atom.GetTotalNumHs())