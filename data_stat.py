import rdkit
from rdkit import Chem
from data_utils import load_data
from tqdm import tqdm


def count_candidate(reactant, product):
    mol1 = Chem.MolFromSmiles(reactant)
    mol2 = Chem.MolFromSmiles(product)

    amap_set1 = set({x.GetAtomMapNum() for x in mol1.GetAtoms()})
    amap_set2 = set({x.GetAtomMapNum() for x in mol2.GetAtoms()})

    return len(amap_set1 - amap_set2)


def count_dist(candidate):
    curr_step, answer = 5, []
    for idx, x in enumerate(candidate):
        if x > curr_step:
            answer.append(idx)
            curr_step += 5

    return answer


if __name__ == '__main__':
    train_rec, train_prod, _ = load_data('../data/UTPSO-50K', 'train')
    valid_rec, valid_prod, _ = load_data('../data/UTPSO-50K', 'val')
    test_rec, test_prod, _ = load_data('../data/UTPSO-50K', 'test')

    train_candidate = [
        count_candidate(x, train_prod[idx])
        for idx, x in enumerate(tqdm(train_rec))
    ]

    valid_candidate = [
        count_candidate(x, valid_prod[idx])
        for idx, x in enumerate(tqdm(valid_rec))
    ]

    test_candidate = [
        count_candidate(x, test_prod[idx])
        for idx, x in enumerate(tqdm(test_rec))
    ]

    candidate = train_candidate + valid_candidate + test_candidate

    train_candidate.sort()
    valid_candidate.sort()
    test_candidate.sort()
    candidate.sort()

    cnt = {
        'train': count_dist(train_candidate),
        'valid': count_dist(valid_candidate),
        'test': count_dist(test_candidate),
        'total': count_dist(candidate)
    }


    print('+-------' + '+-------' * 7 + '+')
    print('|       ', end='')
    for i in range(7):
    	print("|{:^7}".format((i + 1) * 5), end='')

    print('|\n+-------' + '+-------' * 7 + '+')

    print('|{:^7}|'.format('train'), end='')
    for i in range(7):
    	if i >= len(cnt['train']):
    		print('100.00%|', end='')
    	else:
    		print('{:>6.2f}%|'.format(cnt['train'][i] * 100 / len(train_rec)), end='')

    print('\n|{:^7}|'.format('valid'), end='')
    for i in range(7):
    	if i >= len(cnt['valid']):
    		print('100.00%|', end='')
    	else:
    		print('{:>6.2f}%|'.format(cnt['valid'][i] * 100 / len(valid_rec)), end='')

    print('\n|{:^7}|'.format('test'), end='')
    for i in range(7):
    	if i >= len(cnt['test']):
    		print('100.00%|', end='')
    	else:
    		print('{:>6.2f}%|'.format(cnt['test'][i] * 100 / len(test_rec)), end='')
    print('\n+-------' + '+-------' * 7 + '+')



