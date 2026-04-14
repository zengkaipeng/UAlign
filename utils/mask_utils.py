import torch


def generate_square_subsequent_mask(sz, device='cpu'):
    mask = (torch.triu(torch.ones((sz, sz))) == 1).transpose(0, 1)
    return (mask == 0).to(device)
