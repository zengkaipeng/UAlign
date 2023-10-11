from Dataset import OverallDataset, overall_col_fn
from data_utils import load_data, create_overall_dataset

if __name__ == '__main__':
    rec, prod, rxn = load_data(r'..\data\UTPSO-50K', 'val')
    dataset = create_overall_dataset(
        rec[:100], prod[:100], pad_num=10, rxn_class=None, kekulize=False
    )

    print(dataset)
