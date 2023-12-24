import torch
from data_utils import (
    generate_square_subsequent_mask,
    convert_log_into_label, avg_edge_logs
)
from utils.chemistry_parse import (
    canonical_smiles, add_random_Amap, edit_to_synthons,
    get_mol, get_bond_info, BOND_FLOAT_TYPES
)

from tokenlizer import smi_tokenizer

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


def beam_seach_one(
    smiles, model, tokenizer, device, beam_size=10, rxn=None,
    start_token='<CLS>', end_token='<END>', sep_token='`',
):
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

    for state, syn, score in topk_synthons:
        syn_tokens = [start_token]
        syn_tokens.extend(smi_tokenizer(syn.replace('.', sep_token)))
        syn_tokens.append(end_token)
        xip = tokenizer.encode2d([syn_tokens])
        memory, mem_pad = model.encode(
            xip, node_emb, prod_graph.batch_mask,
            graph_rxn=rxn, trans_ip_key_padding=None
        )
        


def beam_search_lg(
    b_memory, b_memory_pad, num_sep, model, tokenizer, device, max_len,
    size=2, begin_token='<CLS>', end_token='<END>', sep_token='`'
):
    model = model.eval()
    end_id = tokenizer.token2idx[end_token]
    beg_id = tokenizer.token2idx[begin_token]
    tgt = torch.LongTensor([[beg_id]]).to(device)
    probs = torch.Tensor([0]).to(device)
    lens = torch.Tensor([0]).to(device)
    alive = torch.Tensor([1]).to(device).bool()
    n_spe = torch.Tensor([0]).to(device)
    sep_id = tokenizer.token2idx[sep_token]

    with torch.no_grad():
        for idx in range(max_len):
            input_beam, prob_beam = [], []
            alive_beam, len_beam, sep_beam = [], [], []
            ended = torch.logical_not(alive)
            if torch.any(ended).item():
                tgt_pad = torch.ones_like(tgt[ended, :1]).long()
                tgt_pad = tgt_pad.to(device) * end_id
                input_beam.append(torch.cat([tgt[ended], tgt_pad], dim=-1))
                prob_beam.append(probs[ended])
                alive_beam.append(alive[ended])
                len_beam.append(lens[ended])
                sep_beam.append(sep_beam[ended])

            if torch.all(ended).item():
                break

            tgt = tgt[alive]
            real_size = min(tgt.shape[0], size)
            memory = b_memory.repeat(real_size, 1, 1)
            mem_pad_mask = b_mem_pad_mask.repeat(real_size, 1)
            tgt_mask = generate_square_subsequent_mask(tgt.shape[1])
            tgt_mask = tgt_mask.to(device)
            result = model.decode(
                tgt=tgt, memory=memory, mem_pad=mem_pad_mask,
                trans_op_mask=tgt_mask,
            )
            result = torch.log_softmax(result[:, -1], dim=-1)
            result_top_k = result.topk(size, dim=-1, largest=True, sorted=True)

            for tdx, ep in enumerate(result_top_k.values):
                not_end = result_top_k.indices[tdx] != end_id
                is_sep = result_top_k.indices[tdx] == sep_id
                tgt_base = tgt[tdx].repeat(size, 1)
                this_seq = result_top_k.indices[tdx].unsqueeze(-1)
                tgt_base = torch.cat([tgt_base, this_seq], dim=-1)
                input_beam.append(tgt_base)
                prob_beam.append(ep + probs[tdx])
                alive_beam.append(not_end)
                len_beam.append(torch.ones(size).long().to(device) * (idx + 1))
                sep_beam.append(is_sep + n_spe[tdx])

            input_beam = torch.cat(input_beam, dim=0)
            prob_beam = torch.cat(prob_beam, dim=0)
            alive_beam = torch.cat(alive_beam, dim=0)
            len_beam = torch.cat(len_beam, dim=0)
            sep_beam = torch.cat(sep_beam, dim=0)

            illegal = (~alive_beam) & (sep_beam != num_sep)
            prob_beam[illegal] = -2e9
            # num leaving group mismatch num synthons

            beam_top_k = prob_beam.topk(size, dim=0, largest=True, sorted=True)
            tgt = input_beam[beam_top_k.indices]
            probs = beam_top_k.values
            alive = alive_beam[beam_top_k.indices]
            lens = len_beam[beam_top_k.indices]
            n_spe = sep_beam[beam_top_k.indices]

    answer = [(probs[idx].item(), t.tolist()) for idx, t in enumerate(tgt)]
    answer.sort(reverse=True)
    real_answer, real_prob = [], []
    for y, x in answer[:size]:
        r_smiles = tokenizer.decode1d(x)
        r_smiles = r_smiles.replace(end_token, "").replace(begin_token, "")
        if get_mol(r_smiles.replace(sep_token, '.')) is None:
            continue
        real_answer.append(r_smiles)
        real_prob.append(y)
    return real_answer, real_prob
