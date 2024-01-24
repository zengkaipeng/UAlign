from tokenlizer import smi_tokenizer
from utils.chemistry_parse import canonical_smiles, clear_map_number
import os
import pandas
from tqdm import tqdm
import json
from rdkit import Chem
import sys
import multiprocessing

def get_all_smiles(mol):
    atms, answer = [x.GetIdx() for x in mol.GetAtoms()], []
    for x in atms:
        px = Chem.MolToSmiles(mol, rootedAtAtom=x, canonical=True)
        answer.append(px)
    return answer


def get_tokens(pid, rxns, Q):
    all_tokens = set()
    for idx, x in enumerate(rxns):
        reac, prod = x.split('>>')
        prod_mol = Chem.MolFromSmiles(clear_map_number(prod))
        for y in get_all_smiles(prod_mol):
            all_tokens.update(smi_tokenizer(y))

        for y in reac.split('.'):
            reac_mol = Chem.MolFromSmiles(clear_map_number(y))
            for z in get_all_smiles(reac_mol):
                all_tokens.update(smi_tokenizer(z))

        all_tokens.update(smi_tokenizer(clear_map_number(reac)))

        if (idx + 1) % 5000 == 0:
            print(f'[Pid {pid}] {idx + 1} / {len(rxns)}')
    Q.put(all_tokens)


if __name__ == '__main__':
    if len(sys.argv) < 3 or sys.argv[1] == '--help':
        print(
            'generate tokenizer for given files and the last '
            'position should be the output file for tokens'
        )

    output_file, all_tokens = sys.argv[-1], set()
    batch_step, num_proc = 20000, os.cpu_count() - 2


    pool = multiprocessing.Pool(processes=num_proc)
    MpQ = multiprocessing.Manager().Queue()

    ppargs, pid_cnt = [], 0
    for x in sys.argv[1: -1]:
        x_file = pandas.read_csv(x)
        all_data = x_file['reactants>reagents>production'].tolist()
        for idx in range(0, len(all_data), batch_step):
            ppargs.append([pid_cnt, all_data[idx: idx + batch_step], MpQ])
            pid_cnt += 1



    pool.starmap(get_tokens, ppargs)
    pool.close()
    pool.join()

    while not MpQ.empty():
        all_tokens.update(MpQ.get())

    with open(output_file, 'w') as Fout:
        json.dump(list(all_tokens), Fout, indent=4)
