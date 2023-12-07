import torch
from data_utils import generate_square_subsequent_mask
from utils.chemistry_parse import canonical_smiles


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
    return canonical_smiles(answer)


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
