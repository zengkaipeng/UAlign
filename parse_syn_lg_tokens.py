import rdkit
from rdkit import Chem
from utils.chemistry_parse import (
    clear_map_number, canonical_smiles, get_synthons,
    break_fragements, get_leaving_group, get_all_amap,
    get_mol_belong
)
from draws import mol2svg
from utils.chemistry_parse import add_random_Amap
import argparse
from tokenlizer import smi_tokenizer
import json
from tqdm import tqdm
from data_utils import load_data


def get_synthon_lg(reac, prod):
    deltaH, deltaE = get_synthons(prod, reac)
    break_edges = set()
    for (src, dst), (otype, ntype) in deltaE.items():
        if otype != ntype and ntype == 0:
            break_edges.update([(src, dst), (dst, src)])

    synthon_str = break_fragements(prod, break_edges, canonicalize=True)
    lgs, conn_edgs = get_leaving_group(prod, reac)

    this_reac, belong = reac.split('.'), {}
    for tdx, rx in enumerate(this_reac):
        belong.update({k: tdx for k in get_all_amap(rx)})

    lg_ops = [[] for _ in range(len(this_reac))]

    for x in lgs:
        lg_ops[get_mol_belong(x, belong)].append(x)

    lg_ops = [clear_map_number('.'.join(x)) for x in lg_ops]
    lg_ops = '`'.join(lg_ops)

    return synthon_str, lg_ops


if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('--dir', required=True)
    args = parser.parse_args()

    train_rec, train_prod, train_rxn = load_data(args.dir, 'train')
    val_rec, val_prod, val_rxn = load_data(args.dir, 'val')
    test_rec, test_prod, test_rxn = load_data(args.dir, 'test')

    # print(get_synthon_lg(
    #     '[Br:1][c:2]1[cH:3][cH:4][c:5]([CH:6]=[O:7])[cH:11][cH:12]1.[CH2:8]([CH2:9][OH:10])[OH:13]',
    #     '[Br:1][c:2]1[cH:3][cH:4][c:5]([CH:6]2[O:7][CH2:8][CH2:9][O:10]2)[cH:11][cH:12]1'
    # ))

    syn_tokens, lg_tokens = set(), set()
    train_syns, train_lg = [], []
    for idx, prod in enumerate(tqdm(train_prod)):
        syn, lg = get_synthon_lg(train_rec[idx], prod)
        train_syns.append(syn)
        train_lg.append(lg)
        syn_tokens.update(smi_tokenizer(syn))
        lg_tokens.update(smi_tokenizer(lg))

    with open('train_synthons.txt', 'w') as Fout:
        json.dump(train_syns, Fout, indent=4)

    with open('train_lgs.txt', 'w') as Fout:
        json.dump(train_lg, Fout, indent=4)
