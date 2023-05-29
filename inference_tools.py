import torch
from data_utils import generate_square_subsequent_mask


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
