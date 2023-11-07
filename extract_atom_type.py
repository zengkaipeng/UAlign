from utils.chemistry_parse import get_node_types
from data_utils import load_data
from tqdm import tqdm


def get_all_atom_types(reac_list):
    result = set()
    for mol in tqdm(reac_list):
        result.update(get_node_types(mol, return_idx=False).values())
    return result


if __name__ == '__main__':
    data_path = '../data/UTPSO-50K'
    train_reac, train_prod, _ = load_data(data_path, 'train')
    valid_reac, valid_prod, _ = load_data(data_path, 'val')
    test_reac, test_prod, _ = load_data(data_path, 'test')

    result = set()

    result.update(get_all_atom_types(train_reac))
    result.update(get_all_atom_types(train_prod))
    result.update(get_all_atom_types(valid_reac))
    result.update(get_all_atom_types(valid_prod))
    result.update(get_all_atom_types(test_reac))
    result.update(get_all_atom_types(test_prod))

    print(result)

    type2idx = {val: idx + 1 for idx, val in enumerate(result)}
    print(type2idx)
