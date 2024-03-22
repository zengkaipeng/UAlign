import argparse
from utils.chemistry_parse import canonical_smiles
from rdkit import Chem
import os
import json

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Parser to deal with things')
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='cano_output')
    parser.add_argument('--output_name', type=str, default='')

    args = parser.parse_args()

    if args.output_name == '':
        args.output_name = os.path.basename(args.input)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    outf = os.path.join(args.output_dir, args.output_name)

    with open(args.input) as Fin:
        INFO = json.load(Fin)

    out_answers = []
    for x in INFO['answer']:
        this_q = {'query': x['query'], 'answer': []}
        for idx, y in enumerate(x['answer']):
            if Chem.MolFromSmiles(y) is None:
                this_q['answer'].append((None, x['prob'][idx]))
            else:
                this_q['answer'].append((canonical_smiles(y), x['prob'][idx]))
        out_answers.append(this_q)

    with open(outf, 'w') as Fout:
        json.dump({
            'answer': out_answers,
            'org_file': args.input,
        }, Fout, indent=4)
