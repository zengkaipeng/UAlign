from Dataset import OverallDataset, overall_col_fn
from data_utils import load_data, create_overall_dataset
from torch.utils.data import DataLoader

if __name__ == '__main__':
    # by case

    rec = [
        '[Br:1][c:2]1[cH:3][cH:4][c:5]([CH:6]=[O:7])[cH:11][cH:12]1.[CH2:8]([CH2:9][OH:10])[OH:13]',
        '[CH2:9]([CH2:10][OH:11])[OH:12].[CH3:1][C:2]([c:3]1[cH:4][cH:5][cH:6][o:7]1)=[O:8]',
        '[NH2:3][c:4]1[cH:5][cH:6][c:7]([N+:8](=[O:9])[O-:10])[cH:11][cH:12]1.[O:1]=[C:2]([C:13]([F:14])([F:15])[F:16])[O:19][C:18](=[O:17])[C:20]([F:21])([F:22])[F:23]',
        '[CH3:1][CH:2]([CH3:3])[C:4](=[O:5])[Cl:10].[NH2:6][C:7]([NH2:8])=[S:9]'
    ]
    prod = [
        '[Br:1][c:2]1[cH:3][cH:4][c:5]([CH:6]2[O:7][CH2:8][CH2:9][O:10]2)[cH:11][cH:12]1',
        '[CH3:1][C:2]1([c:3]2[cH:4][cH:5][cH:6][o:7]2)[O:8][CH2:9][CH2:10][O:11]1',
        '[O:1]=[C:2]([NH:3][c:4]1[cH:5][cH:6][c:7]([N+:8](=[O:9])[O-:10])[cH:11][cH:12]1)[C:13]([F:14])([F:15])[F:16]',
        '[CH3:1][CH:2]([CH3:3])[C:4](=[O:5])[NH:6][C:7]([NH2:8])=[S:9]'
    ]

    dataset = create_overall_dataset(rec, prod, rxn_class=None, kekulize=False)
    col_fn = overall_col_fn(selfloop=True, pad_num=10)

    loader = DataLoader(
        dataset, collate_fn=col_fn, shuffle=False, batch_size=2
    )

    print('Loader Done')

    for data in loader:
        # print(data)
        decoder_graph = data[1]
        # print(decoder_graph.attn_mask.long())
        # exit()
        print(decoder_graph.ptr, '\n', decoder_graph.edge_index)

        

    # run all
    rec, prod, rxn = load_data(r'..\data\UTPSO-50K', 'val')
    dataset = create_overall_dataset(
        rec[:100], prod[:100], rxn_class=None, kekulize=False
    )

    print(dataset)

    col_fn = overall_col_fn(selfloop=True, pad_num=10)
