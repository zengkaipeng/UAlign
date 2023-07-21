from tokenlizer import smi_tokenizer
from utils.chemistry_parse import canonical_smiles, clear_map_number
import os
import pandas
from tqdm import tqdm
import json
from rdkit import Chem
import sys
if __name__ == '__main__':
    if len(sys.argv < 3) or sys.argv[1] == '--help':
        print(
            'generate tokenizer for given files and the last '
            'position should be the output file for tokens'
        )

    output_file, all_tokens = sys.argv[-1], set()

    for x in sys.argv[1: -1]:
        x_file = pandas.read_csv(x)
        for rxn in x_file['reactants>reagents>production']:
            reac, prod = rxn.split('>>')
            token_reac = smi_tokenizer(clear_map_number(reac))
            token_prod = smi_tokenizer(clear_map_number(prod))

            all_tokens.update(token_prod)
            all_tokens.update(token_reac)

    with open(output_file, 'w') as Fout:
        json.dump(list(all_tokens), Fout, indent=4)
    