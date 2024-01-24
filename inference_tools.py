import torch
from data_utils import generate_square_subsequent_mask
from rdkit import Chem


def check_valid(smi):
    mol = Chem.MolFromSmiles(smi)
    return mol is not None


def greedy_inference_one(
    model, tokenizer, graph, device, max_len,
    begin_token='<CLS>', end_token='<END>'
):
    model = model.eval()
    tgt = torch.LongTensor([tokenizer.encode1d([begin_token])])
    tgt = tgt.to(device)

    end_id = tokenizer.token2idx[end_token]
    with torch.no_grad():
        memory, mem_pad_mask = model.encode(graph)
        for idx in range(max_len):
            tgt_mask = generate_square_subsequent_mask(tgt.shape[1])
            tgt_mask = tgt_mask.to(device)
            result = model.decode(
                tgt=tgt, memory=memory, tgt_mask=tgt_mask,
                memory_padding_mask=mem_pad_mask
            )
            result = result[:, -1].argmax(dim=-1)
            # [1, 1]
            if result.item() == end_id:
                break
            tgt = torch.cat([tgt, result.unsqueeze(-1)], dim=-1)

    tgt_list = tgt.tolist()[0]
    answer = tokenizer.decode1d(tgt_list)
    answer = answer.replace(end_token, "").replace(begin_token, "")
    return answer


def beam_search_one(
    model, tokenizer, graph, device, max_len, size=2, pen_para=0,
    begin_token='<CLS>', end_token='<END>',  validate=False
):
    model = model.eval()
    end_id = tokenizer.token2idx[end_token]
    beg_id = tokenizer.token2idx[begin_token]
    tgt = torch.LongTensor([[beg_id]]).to(device)
    probs = torch.Tensor([0]).to(device)
    lens = torch.Tensor([0]).to(device)
    alive = torch.Tensor([1]).to(device).bool()

    n_close = torch.Tensor([0]).to(device)
    fst_idx = tokenizer.token2idx['(']
    sec_idx = tokenizer.token2idx[")"]

    with torch.no_grad():
        base_memory, base_mem_pad_mask = model.encode(graph)
        for idx in range(max_len):
            input_beam, prob_beam = [], []
            alive_beam, len_beam, col_beam = [], [], []

            ended = torch.logical_not(alive)
            if torch.any(ended).item():
                tgt_pad = torch.ones_like(tgt[ended, :1]).long()
                tgt_pad = tgt_pad.to(device) * end_id
                input_beam.append(torch.cat([tgt[ended], tgt_pad], dim=-1))

                prob_beam.append(probs[ended])
                alive_beam.append(alive[ended])
                len_beam.append(lens[ended])
                col_beam.append(n_close[ended])

            if torch.all(ended).item():
                break

            tgt = tgt[alive]
            probs = probs[alive]
            lens = lens[alive]
            n_close = n_close[alive]

            real_size = min(tgt.shape[0], size)
            memory = base_memory.repeat(real_size, 1, 1)
            mem_pad_mask = base_mem_pad_mask.repeat(real_size, 1)
            tgt_mask = generate_square_subsequent_mask(tgt.shape[1])
            tgt_mask = tgt_mask.to(device)
            result = model.decode(
                tgt=tgt, memory=memory, tgt_mask=tgt_mask,
                memory_padding_mask=mem_pad_mask
            )
            result = torch.log_softmax(result[:, -1], dim=-1)
            result_top_k = result.topk(size, dim=-1, largest=True, sorted=True)

            for tdx, ep in enumerate(result_top_k.values):
                not_end = result_top_k.indices[tdx] != end_id

                is_fst = result_top_k.indices[tdx] == fst_idx
                is_sed = result_top_k.indices[tdx] == sec_idx

                tgt_base = tgt[tdx].repeat(size, 1)
                this_seq = result_top_k.indices[tdx].unsqueeze(-1)
                tgt_base = torch.cat([tgt_base, this_seq], dim=-1)
                input_beam.append(tgt_base)
                prob_beam.append(ep + probs[tdx])
                alive_beam.append(not_end)
                len_beam.append(torch.ones(size).long().to(device) * (idx + 1))
                col_beam.append(1. * is_fst - 1. * is_sed + n_close[tdx])

            input_beam = torch.cat(input_beam, dim=0)
            prob_beam = torch.cat(prob_beam, dim=0)
            alive_beam = torch.cat(alive_beam, dim=0)
            len_beam = torch.cat(len_beam, dim=0)
            col_beam = torch.cat(col_beam, dim=0)

            illegal = (col_beam < 0) | ((~alive_beam) & (col_beam != 0))
            prob_beam[illegal] = -2e9

            # ") num" > "( num"
            # the str ends but () not close

            if 0 < pen_para < 1:
                prob_beam /= (len_beam ** pen_para)

            beam_top_k = prob_beam.topk(size, dim=0, largest=True, sorted=True)
            tgt = input_beam[beam_top_k.indices]
            probs = beam_top_k.values
            alive = alive_beam[beam_top_k.indices]
            lens = len_beam[beam_top_k.indices]
            n_close = col_beam[beam_top_k.indices]

    answer = [(probs[idx].item(), t.tolist()) for idx, t in enumerate(tgt)]
    answer.sort(reverse=True)
    real_answer, real_prob = [], []
    for y, x in answer[:size]:
        r_smiles = tokenizer.decode1d(x)
        r_smiles = r_smiles.replace(end_token, "").replace(begin_token, "")
        r_smiles = r_smiles.replace('<UNK>', '')
        if validate and not check_valid(r_smiles):
            continue
        real_answer.append(r_smiles)
        real_prob.append(y)
    return real_answer, real_prob
