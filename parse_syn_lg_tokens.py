import rdkit
from rdkit import Chem
from utils.chemistry_parse import (
    clear_map_number, canonical_smiles, get_synthon_edits,
    break_fragements, get_all_amap, get_leaving_group_synthon,
    get_mol_belong, edit_to_synthons
)
from draws import mol2svg
from utils.chemistry_parse import add_random_Amap
import argparse
from tokenlizer import smi_tokenizer
import json
from tqdm import tqdm
from data_utils import load_data
import os


def get_synthon_lg(reac, prod, consider_inner=False):

    _, _, _, deltE = get_synthon_edits(reac, prod, consider_inner)

    # print('[SYN]')
    synthon_str = edit_to_synthons(prod, {k: v[1] for k, v in deltE.items()})
    synthon_str = clear_map_number(synthon_str)
    synthon_str = '`'.join(synthon_str.split('.'))
    # print('[LG]')
    lgs, syns, conn_edges = get_leaving_group_synthon(
        prod, reac, consider_inner_bonds=consider_inner
    )

    this_reac, belong = reac.split('.'), {}
    for tdx, rx in enumerate(this_reac):
        belong.update({k: tdx for k in get_all_amap(rx)})

    lg_ops = [[] for _ in range(len(this_reac))]
    syn_ops = [[] for _ in range(len(this_reac))]

    for x in lgs:
        lg_ops[get_mol_belong(x, belong)].append(x)

    for x in syns:
        syn_ops[get_mol_belong(x, belong)].append(x)

    lg_ops = [clear_map_number('.'.join(x)) for x in lg_ops]
    lg_ops = '`'.join(lg_ops)

    syn_ops = [clear_map_number('.'.join(x)) for x in syn_ops]
    syn_ops = '`'.join(syn_ops)

    return synthon_str, syn_ops, lg_ops


def update_info(reac, prod, syns, lgs, syn_toks, lg_toks):
    syn_str, syn_ops, lg_ops = get_synthon_lg(reac, prod, False)

    syns.append(syn_str)
    lgs.append(lg_ops)

    syn_tokens.update(smi_tokenizer(syn_str))
    syn_tokens.update(smi_tokenizer(syn_ops))
    syn_tokens.update(smi_tokenizer(lg_ops))

    syn_str, syn_ops, lg_ops = get_synthon_lg(reac, prod, True)

    syn_tokens.update(smi_tokenizer(syn_str))
    syn_tokens.update(smi_tokenizer(syn_ops))
    lg_tokens.update(smi_tokenizer(lg_ops))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('--dir', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    if not os.path.exists(args.output):
        os.makedirs(args.output)

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
        update_info(
            reac=train_rec[idx], prod=prod, syns=train_syns,
            lgs=train_lg, syn_toks=syn_tokens, lg_toks=lg_tokens
        )

    with open(os.path.join(args.output, 'train_synthons.txt'), 'w') as Fout:
        json.dump(train_syns, Fout, indent=4)

    with open(os.path.join(args.output, 'train_lgs.txt'), 'w') as Fout:
        json.dump(train_lg, Fout, indent=4)

    val_syns, val_lg = [], []
    for idx, prod in enumerate(tqdm(val_prod)):
        update_info(
            reac=val_rec[idx], prod=prod, syns=val_syns,
            lgs=val_lg, syn_toks=syn_tokens, lg_toks=lg_tokens
        )

    with open(os.path.join(args.output, 'val_synthons.txt'), 'w') as Fout:
        json.dump(val_syns, Fout, indent=4)

    with open(os.path.join(args.output, 'val_lgs.txt'), 'w') as Fout:
        json.dump(val_lg, Fout, indent=4)

    test_syns, test_lg = [], []
    for idx, prod in enumerate(tqdm(test_prod)):
        update_info(
            reac=test_rec[idx], prod=prod, syns=test_syns,
            lgs=test_lg, syn_toks=syn_tokens, lg_toks=lg_tokens
        )

    with open(os.path.join(args.output, 'test_synthons.txt'), 'w') as Fout:
        json.dump(test_syns, Fout, indent=4)

    with open(os.path.join(args.output, 'test_lgs.txt'), 'w') as Fout:
        json.dump(test_lg, Fout, indent=4)

    with open(os.path.join(args.output, 'synthon_token.json'), 'w') as Fout:
        json.dump(list(syn_tokens), Fout)

    with open(os.path.join(args.output, 'lg_tokens.json'), 'w') as Fout:
        json.dump(list(lg_tokens), Fout)

    with open(os.path.join(args.output, 'all_token.json'), 'w') as Fout:
        json.dump(list(lg_tokens | syn_tokens), Fout)

    print('[SYN TOKEN]', len(syn_tokens))
    print('[LG TOKEN]', len(lg_tokens))
