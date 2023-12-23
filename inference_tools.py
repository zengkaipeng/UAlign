import torch
from data_utils import (
    generate_square_subsequent_mask,
    convert_log_into_label, avg_edge_logs
)
from utils.chemistry_parse import (
    canonical_smiles, add_random_Amap, edit_to_synthons,
    get_mol, get_bond_info, BOND_FLOAT_TYPES
)

from utils.graph_utils import smiles2graph

from Dataset import make_prod_graph
from copy import deepcopy
from queue import PriorityQueue as PQ
import math


class SynthonState:
    def __init__(self, state, score):
        self.state = state
        self.score = score

    def __lt__(self, other):
        return self.score > other.score


def get_topk_synthons(smiles, bond_types, edge_logs, beam_size):
    next_state = {}
    init_state, init_score = {}, 0

    for k, v in edge_logs.items():
        idx = [0, 1, 2, 3, 4]
        idx.sort(key=lambda x: -v[x])
        last_state = None
        for p in idx:
            this_state = (k, p, math.log(v[p] + 1e-9))
            if p == 4 and bond_types[k][0] != 1.5:
                # can not build a aron bond
                continue

            if last_state is not None:
                next_state[last_state[: 2]] = this_state
            else:
                init_state[k] = (p, math.log([p] + 1e-9))
                init_score += math.log(v[p] + 1e-9)

            last_state = this_state

    Q = PQ()

    Q.put(SynthonState(init_state, init_score))

    valid_synthons = []

    while not Q.empty():
        curr_state, curr_score = Q.get()
        delta = {}
        for k, v in curr_state.items():
            old_type = bond_types[k][0]
            new_type = BOND_FLOAT_TYPES[v[0]]
            if old_type != new_type:
                delta[k] = new_type
        try:
            this_syn = edit_to_synthons(smiles, edge_edits)
        except Exception as e:
            this_syn = None

        if this_syn is not None and get_mol(this_syn) is not None:
            valid_synthons.append((curr_state, this_syn, curr_score))

        if len(valid_synthons) == beam_size:
            break

        for k, v in curr_state.items():
            nx_state = next_state.get((k, v[0]), None)
            if nx_state is not None:
                _, nx_idx, nx_score = nx_state
                cur_idx, cur_score = v
                sub_state = deepcopy(curr_state)
                sub_state[k] = (nx_idx, nx_score)
                sub_score = curr_score - cur_score + nx_score
                Q.put(SynthonState(sub_state, sub_score))

    return valid_synthons


def beam_seach(smiles, model, beam_size=10, rxn=None):
    model = model.eval()
    mol = get_mol(smiles, kekulize=False)
    assert mol is not None, "Invalid smiles passed"

    if any(x.GetAtomMapNum() == 0 for x in mol.GetAtoms()):
        smiles = add_random_Amap(smiles)
        mol = get_mol(smiles)

    bond_types = get_bond_info(mol)

    graph, amap = smiles2graph(smiles, with_amap=True, kekulize=False)
    prod_graph = make_prod_graph(graph, rxn=rxn)
    reidx_amap = {v: k for k, v in amap.items()}

    with torch.no_grad():
        AC_logs, edge_logs, node_emb, edge_emb = \
            model.synthon_forward(prod_graph)

    AC_label = convert_log_into_label(AC_logs, mod='sigmoid')
    edge_logs = avg_edge_logs(edge_logs, prod_graph.edge_index)
    amap_edge_logs = {}
    for (a, b), c in t_edge_logs.items():
        amap_a = reidx_amap[a]
        amap_b = reidx_amap[b]
        key_pair = (min(amap_a, amap_b), max(amap_a, amap_b))
        amap_edge_logs[key_pair] = c

    topk_synthons = get_topk_synthons(
        smiles=smiles, bond_types=bond_types,
        edge_logs=amap_edge_logs, beam_size=beam_size
    )
