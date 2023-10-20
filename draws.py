from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D


def rxn2svg(rxn, output_path=None):
    reac, prod = rxn.split('>>')
    mols = [Chem.MolFromSmiles(reac), Chem.MolFromSmiles(prod)]
    img = Draw.MolsToGridImage(
        mols, molsPerRow=1, subImgSize=(400, 400), useSVG=True,
        legends=['reactant', 'product']
    )
    if output_path is None:
        return img
    else:
        with open(output_path, 'w') as Fout:
            Fout.write(img)


def mol2svg(mol, output_path=None):
    mol = Chem.MolFromSmiles(mol)
    d2d = rdMolDraw2D.MolDraw2DSVG(400, 300)
    d2d.DrawMolecule(mol)
    d2d.FinishDrawing()

    if output_path is None:
        return d2d.GetDrawingText()
    else:
        with open(output_path, 'w') as Fout:
            Fout.write(d2d.GetDrawingText())


if __name__ == '__main__':
    # rxns = [
    #     '[Br:1][c:2]1[cH:3][cH:4][c:5]([CH:6]=[O:7])[cH:11][cH:12]1.[CH2:8]([CH2:9][OH:10])[OH:13]>>[Br:1][c:2]1[cH:3][cH:4][c:5]([CH:6]2[O:7][CH2:8][CH2:9][O:10]2)[cH:11][cH:12]1',
    #     '[CH2:9]([CH2:10][OH:11])[OH:12].[CH3:1][C:2]([c:3]1[cH:4][cH:5][cH:6][o:7]1)=[O:8]>>[CH3:1][C:2]1([c:3]2[cH:4][cH:5][cH:6][o:7]2)[O:8][CH2:9][CH2:10][O:11]1',
    #     '[NH2:3][c:4]1[cH:5][cH:6][c:7]([N+:8](=[O:9])[O-:10])[cH:11][cH:12]1.[O:1]=[C:2]([C:13]([F:14])([F:15])[F:16])[O:19][C:18](=[O:17])[C:20]([F:21])([F:22])[F:23]>>[O:1]=[C:2]([NH:3][c:4]1[cH:5][cH:6][c:7]([N+:8](=[O:9])[O-:10])[cH:11][cH:12]1)[C:13]([F:14])([F:15])[F:16]',
    #     '[CH3:1][CH:2]([CH3:3])[C:4](=[O:5])[Cl:10].[NH2:6][C:7]([NH2:8])=[S:9]>>[CH3:1][CH:2]([CH3:3])[C:4](=[O:5])[NH:6][C:7]([NH2:8])=[S:9]'
    # ]

    # for idx, rxn in enumerate(rxns):
    #     rxn2svg(rxn, output_path=f'tmp_figs/rxn_{idx}.svg')
    
    # mol = '[Br:1][c:2]1[cH:3][cH:4][c:5]([CH:6]2[O:7][CH2:8][CH2:9][O:10]2)[cH:11][cH:12]1'
    # mol2svg(mol, 'tmp_figs/example1.svg')
    # mol = 'OC(OCCCl)c1ccc(Br)cc1'
    # mol2svg(mol, 'tmp_figs/example2.svg')
    
    rxn = '[CH:1]1=[CH:2][CH2:3][c:4]2[cH:5][cH:6][cH:7][cH:8][c:9]2[CH2:10]1>>[cH:1]1[cH:2][cH:3][c:4]2[cH:5][cH:6][cH:7][cH:8][c:9]2[cH:10]1'
    rxn2svg(rxn, 'tmp_figs/example_rxn.svg')