import torch


def generate_square_subsequent_mask(sz, device='cpu'):
    mask = (torch.triu(torch.ones((sz, sz))) == 1).transpose(0, 1)
    return (mask == 0).to(device)


def generate_tgt_mask(tgt, tokenizer, pad='<PAD>', device='cpu'):
    pad_idx = tokenizer.token2idx[pad]
    tgt_pad_mask = (tgt == pad_idx).to(device)
    tgt_sub_mask = generate_square_subsequent_mask(tgt.shape[1], device)
    return tgt_pad_mask, tgt_sub_mask
