import os
import json
import pandas
from tqdm import tqdm
from tokenlizer import smi_tokenizer
from utils.chemistry_parse import canonical_smiles, clear_map_number
import rdkit
from rdkit import Chem


def check_valid(smi):
    mol = Chem.MolFromSmiles(smi)
    return False if mol is None else True


UTPSO_50_PATH = '../data/UTPSO-50K/'
UTPSO_50_FILES = [f'raw_{x}.csv' for x in ['train', 'val', 'test']] +\
    [f'canonicalized_raw_{x}.csv' for x in ['train', 'val', 'test']]

EXTEND_PATH = '../data/ord/all_data.json'

vocab = set()
for x in UTPSO_50_FILES:
    x_path = os.path.join(UTPSO_50_PATH, x)
    df_train = pandas.read_csv(x_path)
    print(f'[INFO] Processing {x_path}')
    for resu in tqdm(df_train['reactants>reagents>production']):
        rea, prd = resu.strip().split('>>')
        vocab.update(smi_tokenizer(clear_map_number(rea)))
        vocab.update(smi_tokenizer(clear_map_number(prd)))


with open(EXTEND_PATH) as Fin:
    extend_INFO = json.load(Fin)

print(f'[INFO] processing {EXTEND_PATH}')
for k, v in tqdm(extend_INFO.items()):
    reac = '.'.join(x.strip() for x in v['reactant'] if check_valid(x))
    prd = '.'.join(x.strip() for x in v['product'] if check_valid(x))
    reg = '.'.join(x.strip() for x in v['reagent'] if check_valid(x))
    vocab.update(smi_tokenizer(reac))
    vocab.update(smi_tokenizer(prd))
    vocab.update(smi_tokenizer(reg))
    vocab.update(smi_tokenizer(canonical_smiles(reac)))
    vocab.update(smi_tokenizer(canonical_smiles(prd)))
    vocab.update(smi_tokenizer(canonical_smiles(reg)))

vocab |= {'.', '-', '=', '#', '$', ':', '/', '\\'}
vocab |= {str(x) for x in range(10)}

with open('All_token.json', 'w') as Fout:
    json.dump(list(vocab), Fout, indent=4)
