import torch

from Dataset import BinaryEditDataset, edit_col_fn
from data_utils import create_edit_dataset, load_data
from utils.chemistry_parse import get_modified_atoms_bonds
import rdkit
from rdkit import Chem
from rdkit.Chem import Draw

if __name__ == '__main__':
    reac, prod, rxn = load_data('../data/UTPSO-50K', 'val')
    # dataset = create_edit_dataset(reac, prod, rxn_class=rxn, kekulize=False)
    # col_fn = edit_col_fn(selfloop=True)

    # loader = torch.utils.data.DataLoader(
    #     dataset, batch_size=2,
    #     collate_fn=col_fn, shuffle=False
    # )

    for x in range(0, 10, 2):
        mols = [
            Chem.MolFromSmiles(reac[x]), Chem.MolFromSmiles(reac[x + 1]),
            Chem.MolFromSmiles(prod[x]), Chem.MolFromSmiles(prod[x + 1])
        ]

        img = Draw.MolsToGridImage(
            mols, molsPerRow=2, subImgSize=(250, 250), useSVG=True,
            legends=[f'reac_{x}', f'reac_{x + 1}', f'prod_{x}', f'prod_{x + 1}']
        )

        with open(f'data_validation/rxn_{x}-{x + 1}.svg', 'w') as Fout:
            Fout.write(img)

    for i in range(10):
        print(f'[{i}]', get_modified_atoms_bonds(reac[i], prod[i], False))
        exit()





