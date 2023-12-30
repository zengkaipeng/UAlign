from tokenlizer import smi_tokenizer
from utils.chemistry_parse import canonical_smiles, clear_map_number
import os
import pandas
from tqdm import tqdm
import json
from rdkit import Chem
import sys
import multiprocessing


def get_tokens(pid, rxns, Q):
    all_tokens = set()
    for idx, x in enumerate(rxns):
        reac, prod = x.split('>>')
        token_reac = smi_tokenizer(clear_map_number(reac))
        token_prod = smi_tokenizer(clear_map_number(prod))

        all_tokens.update(token_prod)
        all_tokens.update(token_reac)
        if (idx + 1) % 25000 == 0:
            print(f'[Pid {pid}] {idx + 1} / {len(rxns)}')
    Q.put(all_tokens)


if __name__ == '__main__':
    if len(sys.argv) < 3 or sys.argv[1] == '--help':
        print(
            'generate tokenizer for given files and the last '
            'position should be the output file for tokens'
        )

    output_file, all_tokens = sys.argv[-1], set()
    batch_step, num_proc = 100000, os.cpu_count() - 1

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
