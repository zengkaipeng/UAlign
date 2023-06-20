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


def greedy_inference_batch(
    model, tokenizer, graph, device, max_len,
    begin_token='<CLS>', end_token='<END>'
):
    batch_size, model = graph.ptr.shape[0] - 1, model.eval()
    tgt = torch.LongTensor([
        tokenizer.encode1d([begin_token])
    ] * batch_size).to(device)

    end_id = tokenizer.token2idx[end_token]
    alive = torch.ones(batch_size).bool()

    with torch.no_grad():
        memory, mem_pad_mask = model.encode(graph)
        for idx in range(max_len):
            tgt_mask = generate_square_subsequent_mask(tgt.shape[1])
            tgt_mask = tgt_mask.to(device)
            result = model.decode(
                tgt=tgt[alive], memory=memory[alive], tgt_mask=tgt_mask,
                memory_padding_mask=mem_pad_mask[alive]
            )
            result = result[:, -1].argmax(dim=-1)
            padding_result = torch.ones(batch_size, 1) * end_id
            padding_result = padding_result.long().to(device)
            padding_result[alive] = result.unsqueeze(-1)
            tgt = torch.cat([tgt, padding_result], dim=-1)
            alive[alive][result == end_id] = False
            if not torch.any(alive):
                break
    tgt_list = tgt.tolist()
    answer = tokenizer.decode2d(tgt_list)
    return [
        x.replace(end_token, "").replace(begin_token, "")
        for x in answer
    ]


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

    with torch.no_grad():
        base_memory, base_mem_pad_mask = model.encode(graph)
        for idx in range(max_len):
            input_beam, prob_beam = [], []
            alive_beam, len_beam = [], []

            ended = torch.logical_not(alive)
            if torch.any(ended).item():
                tgt_pad = torch.ones_like(tgt[ended, :1]).long()
                tgt_pad = tgt_pad.to(device) * end_id
                input_beam.append(torch.cat([tgt[ended], tgt_pad], dim=-1))

                prob_beam.append(probs[ended])
                alive_beam.append(alive[ended])
                len_beam.append(lens[ended])
            if torch.all(ended).item():
                break

            tgt = tgt[alive]
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
                tgt_base = tgt[tdx].repeat(size, 1)
                this_seq = result_top_k.indices[tdx].unsqueeze(-1)
                tgt_base = torch.cat([tgt_base, this_seq], dim=-1)
                input_beam.append(tgt_base)
                prob_beam.append(ep + probs[tdx])
                alive_beam.append(not_end)
                len_beam.append(torch.ones(size).long().to(device) * (idx + 1))

            input_beam = torch.cat(input_beam, dim=0)
            prob_beam = torch.cat(prob_beam, dim=0)
            alive_beam = torch.cat(alive_beam, dim=0)
            len_beam = torch.cat(len_beam, dim=0)

            if 0 < pen_para < 1:
                prob_beam /= (len_beam ** pen_para)

            beam_top_k = prob_beam.topk(size, dim=0, largest=True, sorted=True)
            tgt = input_beam[beam_top_k.indices]
            probs = beam_top_k.values
            alive = alive_beam[beam_top_k.indices]
            lens = len_beam[beam_top_k.indices]

    answer = [(probs[idx].item(), t.tolist()) for idx, t in enumerate(tgt)]
    answer.sort(reverse=True)
    real_answer, real_prob = [], []
    for y, x in answer[:size]:
        r_smiles = tokenizer.decode1d(x)
        r_smiles = r_smiles.replace(end_token, "").replace(begin_token, "")
        if validate and not check_valid(r_smiles):
            continue
        real_answer.append(r_smiles)
        real_prob.append(y)
    return real_answer, real_prob
