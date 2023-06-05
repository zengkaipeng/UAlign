from tokenlizer import smi_tokenizer
from utils.chemistry_parse import canonical_smiles, clear_map_number
import os
import pandas
from tqdm import tqdm
import json
from rdkit import Chem

def check_tk(smi, F):
    a = smi_tokenizer(smi, False)
    b = smi_tokenizer(smi, True)
    if '.'.join(a) != '.'.join(b):
        print('[INFO] DIFF FOUND', smi)
        F.write(f'[ORG] {smi}\n')
        F.write(f'[TOKOLD] {a}\n')
        F.write(f'[TOKNEW] {b}\n')


def check_valid(smi):
    mol = Chem.MolFromSmiles(smi)
    return False if mol is None else True


Ferr = open("Inequal.txt", 'w')


UTPSO_50_PATH = '../data/UTPSO-50K/'
UTPSO_50_FILES = [f'raw_{x}.csv' for x in ['train', 'val', 'test']] +\
    [f'canonicalized_raw_{x}.csv' for x in ['train', 'val', 'test']]

EXTEND_PATH = '../data/ord/all_data.json'

for x in UTPSO_50_FILES:
    x_path = os.path.join(UTPSO_50_PATH, x)
    df_train = pandas.read_csv(x_path)
    print(f'[INFO] Processing {x_path}')
    for resu in tqdm(df_train['reactants>reagents>production']):
        rea, prd = resu.strip().split('>>')

        check_tk(clear_map_number(rea), Ferr)
        check_tk(clear_map_number(prd), Ferr)


with open(EXTEND_PATH) as Fin:
    extend_INFO = json.load(Fin)


for k, v in tqdm(extend_INFO.items()):
    reac = '.'.join(x.strip() for x in v['reactant'] if check_valid(x))
    prd = '.'.join(x.strip() for x in v['product'] if check_valid(x))
    reg = '.'.join(x.strip() for x in v['reagent'] if check_valid(x))

    check_tk(reac, Ferr)
    check_tk(prd, Ferr)
    check_tk(reg, Ferr)
    check_tk(canonical_smiles(reac), Ferr)
    check_tk(canonical_smiles(prd), Ferr)
    check_tk(canonical_smiles(reg), Ferr)

Ferr.close()
