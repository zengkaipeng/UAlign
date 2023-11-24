from Dataset import OverallDataset, overall_col_fn
from data_utils import load_data, create_overall_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

from draws import rxn2svg

if __name__ == '__main__':

    # run all
    # rec, prod, rxn = load_data(r'..\data\UTPSO-50K', 'train')
    # dataset = create_overall_dataset(
    #     rec, prod, rxn_class=None, kekulize=False
    # )

    # print(dataset)

    # col_fn = overall_col_fn(selfloop=False, pad_num=35)

    # loader = DataLoader(
    #     dataset, collate_fn=col_fn, shuffle=True, batch_size=64
    # )

    # for data in tqdm(loader):
    #     pass

    # exit()

    # by case

    rec = [
        '[CH3:14][C:15]([CH3:16])([CH3:17])[O:18][C:19](=[O:20])[N:1]1[CH2:2][CH2:3][c:4]2[cH:5][cH:6][cH:7][c:8]3[nH:9][cH:10][c:11]([c:12]23)[CH2:13]1',
        '[Br:1][c:2]1[cH:3][cH:4][c:5]([CH:6]=[O:7])[cH:11][cH:12]1.[CH2:8]([CH2:9][OH:10])[OH:13]',
        '[CH2:9]([CH2:10][OH:11])[OH:12].[CH3:1][C:2]([c:3]1[cH:4][cH:5][cH:6][o:7]1)=[O:8]',
        '[NH2:3][c:4]1[cH:5][cH:6][c:7]([N+:8](=[O:9])[O-:10])[cH:11][cH:12]1.[O:1]=[C:2]([C:13]([F:14])([F:15])[F:16])[O:19][C:18](=[O:17])[C:20]([F:21])([F:22])[F:23]',
        '[CH3:1][CH:2]([CH3:3])[C:4](=[O:5])[Cl:10].[NH2:6][C:7]([NH2:8])=[S:9]',
        '[I:17][c:1]1[c:2]([Cl:3])[cH:4][c:5]([O:6][CH3:7])[c:8]([NH2:9])[cH:10]1.[OH:14][B:15]([OH:16])[CH:11]1[CH2:12][CH2:13]1',
    ]
    prod = [
        '[NH:1]1[CH2:2][CH2:3][c:4]2[cH:5][cH:6][cH:7][c:8]3[nH:9][cH:10][c:11]([c:12]23)[CH2:13]1',
        '[Br:1][c:2]1[cH:3][cH:4][c:5]([CH:6]2[O:7][CH2:8][CH2:9][O:10]2)[cH:11][cH:12]1',
        '[CH3:1][C:2]1([c:3]2[cH:4][cH:5][cH:6][o:7]2)[O:8][CH2:9][CH2:10][O:11]1',
        '[O:1]=[C:2]([NH:3][c:4]1[cH:5][cH:6][c:7]([N+:8](=[O:9])[O-:10])[cH:11][cH:12]1)[C:13]([F:14])([F:15])[F:16]',
        '[CH3:1][CH:2]([CH3:3])[C:4](=[O:5])[NH:6][C:7]([NH2:8])=[S:9]',
        '[c:1]1([CH:11]2[CH2:12][CH2:13]2)[c:2]([Cl:3])[cH:4][c:5]([O:6][CH3:7])[c:8]([NH2:9])[cH:10]1',
    ]

    for idx, reac in enumerate(rec):
        rxn2svg(f'{reac}>>{prod[idx]}', f'tmp_figs/rxn_{idx}.svg')

    dataset = create_overall_dataset(rec, prod)
    print(dataset)
