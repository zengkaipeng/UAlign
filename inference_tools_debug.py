import torch
from data_utils import (
    generate_square_subsequent_mask,
    convert_log_into_label, avg_edge_logs
)
from utils.chemistry_parse import (
    canonical_smiles, add_random_Amap, edit_to_synthons,
    get_mol, get_bond_info, BOND_FLOAT_TYPES, add_random_Amap_lg,
    get_all_amap, clear_map_number, get_reactants_from_edits,
    run_special_case
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
                init_state[k] = (p, math.log(v[p] + 1e-9))
                init_score += math.log(v[p] + 1e-9)

            last_state = this_state

    Q = PQ()

    Q.put(SynthonState(init_state, init_score))

    valid_synthons = []

    while not Q.empty():
        htop = Q.get()
        curr_state, curr_score = htop.state, htop.score
        delta = {}
        for k, v in curr_state.items():
            old_type = bond_types[k][0]
            new_type = BOND_FLOAT_TYPES[v[0]]
            if old_type != new_type:
                delta[k] = new_type
        try:
            this_syn = edit_to_synthons(smiles, edge_edits=delta)
        except Exception as e:
            this_syn = None

        if this_syn is not None and get_mol(this_syn) is not None:
            valid_synthons.append((curr_state, delta, this_syn, curr_score))

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


def get_topk_conn(conn_edges, conn_logs, K):
    next_state, init_state, init_score = {}, {}, 0
    for idx, eg in enumerate(conn_edges):
        a, b = eg.tolist()
        logs = conn_logs[idx].tolist()
        idx = [0, 1, 2, 3]
        idx.sort(key=lambda x: -logs[x])
        last_state = None
        for p in idx:
            this_state = (a, b, p, math.log(logs[p] + 1e-9))
            if last_state is not None:
                next_state[last_state[:3]] = this_state
            else:
                init_state[(a, b)] = (p, math.log(logs[p] + 1e-9))
                init_score += math.log(logs[p] + 1e-9)
            last_state = this_state

    Q = PQ()
    Q.put(SynthonState(init_state, init_score))

    result = []
    while not Q.empty():
        htop = Q.get()
        curr_state, curr_score = htop.state, htop.score
        conns = {k: BOND_FLOAT_TYPES[v[0]] for k, v in curr_state.items()}

        result.append((conns, curr_score))

        if len(result) == K:
            break

        for k, v in curr_state.items():
            nx_state = next_state.get((k[0], k[1], v[0]), None)
            if nx_state is not None:
                _, _, nx_idx, nx_score = nx_state
                cur_idx, cur_score = v
                sub_state = deepcopy(curr_state)
                sub_state[k] = (nx_idx, nx_score)
                sub_score = curr_score - cur_score + nx_score
                Q.put(SynthonState(sub_state, sub_score))

    return result


def beam_seach_one(
    smiles, model, tokenizer, device, beam_size=10, rxn=None,
    start_token='<CLS>', end_token='<END>', sep_token='`',
    max_len=100,
):
    model = model.eval()
    mol = get_mol(smiles, kekulize=False)
    assert mol is not None, "Invalid smiles passed"

    if any(x.GetAtomMapNum() == 0 for x in mol.GetAtoms()):
        smiles = add_random_Amap(smiles)
        mol = get_mol(smiles)

    bond_types = get_bond_info(mol)

    graph, amap = smiles2graph(smiles, with_amap=True, kekulize=False)
    prod_graph = make_prod_graph(graph, rxn=rxn).to(device)
    reidx_amap = {v: k for k, v in amap.items()}

    with torch.no_grad():
        AC_logs, edge_logs, node_emb, edge_emb = \
            model.synthon_forward(prod_graph)

    AC_label = convert_log_into_label(AC_logs, mod='sigmoid')

    charge_atoms = {
        reidx_amap[idx] for idx, p in enumerate(AC_label.tolist()) if p > 0
    }

    edge_logs = avg_edge_logs(edge_logs, prod_graph.edge_index, mod='softmax')
    amap_edge_logs = {}
    for (a, b), c in edge_logs.items():
        amap_a = reidx_amap[a]
        amap_b = reidx_amap[b]
        key_pair = (min(amap_a, amap_b), max(amap_a, amap_b))
        amap_edge_logs[key_pair] = c

    topk_synthons = get_topk_synthons(
        smiles=smiles, bond_types=bond_types,
        edge_logs=amap_edge_logs, beam_size=beam_size
    )

    x_beams = []

    for state, delta, syn, score in topk_synthons:
        sorted_syn = syn.split('.')
        cano_syn = [clear_map_number(x) for x in sorted_syn]
        cano_idx = list(range(len(cano_syn)))
        cano_idx.sort(key=lambda x: len(cano_syn[x]))

        cano_syn = [cano_syn[x] for x in cano_idx]
        sorted_syn = '.'.join([sorted_syn[x] for x in cano_idx])

        syn_tokens = [start_token]
        syn_tokens.extend(smi_tokenizer(sep_token.join(cano_syn)))
        syn_tokens.append(end_token)
        xip = tokenizer.encode2d([syn_tokens])
        xip = torch.LongTensor(xip).to(device)

        memory, mem_pad = model.encode(
            xip, node_emb, prod_graph.batch_mask,
            graph_rxn=rxn, trans_ip_key_padding=None
        )

        num_sep = len(cano_syn) - 1

        lgs_with_score = beam_search_lg(
            b_memory=memory, b_memory_pad=mem_pad, num_sep=num_sep,
            model=model, tokenizer=tokenizer, device=device, max_len=max_len,
            size=beam_size, begin_token=start_token, end_token=end_token,
            sep_token=sep_token
        )

        for p, q in lgs_with_score:
            x_beams.append((state, delta, sorted_syn, p, q + score))

    x_beams.sort(key=lambda x: -x[-1])
    topk_syn_lg = x_beams[:beam_size]

    x_beams = []
    for state, delta, syn, lg, score in topk_syn_lg:
        syn_split = syn.split('.')
        lg = add_random_Amap_lg(lg, sep_token)
        lg_split = lg.split(sep_token)
        if all(x == '' for x in lg_split):
            x_beams.append((state, delta, '', {}, score))
            continue

        assert len(syn_split) == len(lg_split), 'Invalid generation, bug find'

        lg_for_mol = '.'.join(x for x in lg.split(sep_token) if x != '')

        lg_graph, lg_amap = smiles2graph(
            lg_for_mol, kekulize=False, with_amap=True
        )

        lg_reidx_amap = {v: k for k, v in lg_amap.items()}

        lg_graph_ip = make_prod_graph(lg_graph, rxn=rxn).to(device)

        with torch.no_grad():
            lg_n_feat, lg_e_feat = model.GNN(lg_graph_ip)

        conn_cog = []
        for idx, syx in enumerate(syn_split):
            lgx = lg_split[idx]
            amap_syn = get_all_amap(syx)
            amap_lg = get_all_amap(lgx)
            for a in amap_syn:
                for b in amap_lg:
                    conn_cog.append((amap[a], lg_amap[b]))

        conn_cog = torch.LongTensor(conn_cog).to(device)
        with torch.no_grad():
            conn_logs, conn_mask = model.eval_conn_forward(
                prod_feat=node_emb, lg_feat=lg_n_feat, conn_edges=conn_cog,
                prod_batch_mask=prod_graph.batch_mask,
                lg_batch_mask=lg_graph_ip.batch_mask
            )
            conn_logs = torch.softmax(conn_logs, dim=-1)

        topk_conns = get_topk_conn(conn_cog[conn_mask], conn_logs, beam_size)

        for conn, conn_score in topk_conns:
            conn = {
                (reidx_amap[a], lg_reidx_amap[b]): v
                for (a, b), v in conn.items() if v != 0
            }
            x_beams.append((
                state, delta, lg_for_mol, conn, score + conn_score
            ))

    x_beams.sort(key=lambda x: -x[-1])

    results = []

    for state, delta, lg, conn, score in x_beams:
        try:
            reactants = get_reactants_from_edits(
                prod_smi=smiles, edge_edits=delta, lgs=lg, conns=conn
            )
            reactants = run_special_case(reactants, charge_atoms)
        except Exception as e:
            reactants = None
            print("error", e)

        if reactants is not None:
            results.append((clear_map_number(reactants), score))

        if len(results) == beam_size:
            break

    return results


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
    alive = torch.BoolTensor([True]).to(device)
    n_spe = torch.Tensor([0]).to(device)
    sep_id = tokenizer.token2idx[sep_token]
    n_close = torch.Tensor([0]).to(device)
    fst_idx = tokenizer.token2idx['(']
    sec_idx = tokenizer.token2idx[")"]

    with torch.no_grad():
        for idx in range(max_len):
            input_beam, prob_beam, clo_beam = [], [], []
            alive_beam, len_beam, sep_beam = [], [], []
            ended = torch.logical_not(alive)
            if torch.any(ended).item():
                tgt_pad = torch.ones_like(tgt[ended, :1]).long()
                tgt_pad = tgt_pad.to(device) * end_id
                input_beam.append(torch.cat([tgt[ended], tgt_pad], dim=-1))
                prob_beam.append(probs[ended])
                alive_beam.append(alive[ended])
                len_beam.append(lens[ended])
                sep_beam.append(n_spe[ended])
                clo_beam.append(n_close[ended])

            if torch.all(ended).item():
                break

            tgt = tgt[alive]
            probs = probs[alive]
            lens = lens[alive]
            n_spe = n_spe[alive]
            n_close = n_close[alive]

            real_size = min(tgt.shape[0], size)
            memory = b_memory.repeat(real_size, 1, 1)
            mem_pad_mask = b_memory_pad.repeat(real_size, 1)
            tgt_mask = generate_square_subsequent_mask(tgt.shape[1])
            tgt_mask = tgt_mask.to(device)
            result = model.decode(
                tgt=tgt, memory=memory, memory_pad=mem_pad_mask,
                trans_op_mask=tgt_mask,
            )
            result = torch.log_softmax(result[:, -1], dim=-1)
            result_top_k = result.topk(size, dim=-1, largest=True, sorted=True)

            for tdx, ep in enumerate(result_top_k.values):
                not_end = result_top_k.indices[tdx] != end_id
                is_sep = result_top_k.indices[tdx] == sep_id

                is_fst = result_top_k.indices[tdx] == fst_idx
                is_sed = result_top_k.indices[tdx] == sec_idx

                tgt_base = tgt[tdx].repeat(size, 1)
                this_seq = result_top_k.indices[tdx].unsqueeze(-1)
                tgt_base = torch.cat([tgt_base, this_seq], dim=-1)
                input_beam.append(tgt_base)
                prob_beam.append(ep + probs[tdx])
                alive_beam.append(not_end)
                len_beam.append(torch.ones(size).long().to(device) * (idx + 1))
                sep_beam.append(is_sep + n_spe[tdx])
                clo_beam.append(1. * is_fst - 1. * is_sed + n_close[tdx])

            input_beam = torch.cat(input_beam, dim=0)
            prob_beam = torch.cat(prob_beam, dim=0)
            alive_beam = torch.cat(alive_beam, dim=0)
            len_beam = torch.cat(len_beam, dim=0)
            sep_beam = torch.cat(sep_beam, dim=0)
            clo_beam = torch.cat(clo_beam, dim=0)

            illegal = (~alive_beam) & (sep_beam != num_sep)
            illegal |= (clo_beam < 0) | (sep_beam > num_sep)
            illegal |= (~alive_beam) & (clo_beam != 0)

            prob_beam[illegal] = -2e9
            # num leaving group mismatch num synthons
            # ") num" > "( num"
            # the str ends but () not close

            beam_top_k = prob_beam.topk(size, dim=0, largest=True, sorted=True)
            tgt = input_beam[beam_top_k.indices]
            probs = beam_top_k.values
            alive = alive_beam[beam_top_k.indices]
            lens = len_beam[beam_top_k.indices]
            n_spe = sep_beam[beam_top_k.indices]
            n_close = clo_beam[beam_top_k.indices]

            print(f'------------{idx}-------------')
            for tdx, p in enumerate(tgt):
                print(tokenizer.decode1d(p.tolist()), n_close[tdx].item(), alive[tdx].item(), probs[tdx].item())

    answer = [
        (probs[idx].item(), t.tolist()) for idx, t in enumerate(tgt[ended])
    ]
    answer.sort(reverse=True)
    real_answer = []
    for y, x in answer[:size]:
        r_smiles = tokenizer.decode1d(x)
        r_smiles = r_smiles.replace(end_token, "").replace(begin_token, "")
        r_for_mol = '.'.join(x for x in r_smiles.split(sep_token) if x != '')

        print('r_for_mol', r_for_mol, y)

        if get_mol(r_for_mol) is None:
            continue
        real_answer.append((r_smiles, y))

    return real_answer
