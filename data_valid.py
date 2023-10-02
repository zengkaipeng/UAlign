from data_utils import load_ext_data
from model import get_col_fc
from model import OnFlyDataset
from torch.utils.data import DataLoader

if __name__ == '__main__':
    data_path = r'..\\data\\UTPSO-50K\\'
    col_fn = get_col_fc(self_loop=True)
    train_rec, train_prod, train_rxn, train_target =\
        load_ext_data(data_path, 'train')


    train_set = OnFlyDataset(
        prod_sm=train_prod, reat_sm=train_rec, target=train_target,
        aug_prob=0.5, randomize=True,
    )

    train_loader = DataLoader(
        train_set, collate_fn=col_fn,
        batch_size=2, shuffle=True,
    )

    for idx, x in enumerate(train_loader):
        ip, tgt = x
        print(ip.batch)
        print(ip.batch_mask)
        print(ip.edge_index.T)
        print(ip.org_mask)
        print(ip.self_mask)
        print(ip.edge_index.shape)
        print(ip.edge_label.shape)
        print('--------------------')
        if idx == 5:
            break