from typing import List

import torch
from rdkit import Chem


def check_valid(smi):
    mol = Chem.MolFromSmiles(smi)
    return mol is not None


def _strip_tokens(smi, begin_token, end_token, pad_token):
    smi = smi.replace(begin_token, '')
    smi = smi.replace(end_token, '')
    smi = smi.replace(pad_token, '')
    smi = smi.replace('<UNK>', '')
    return smi


def _pack_beam_answers(
    seqs: torch.Tensor,
    scores: torch.Tensor,
    belong: torch.Tensor,
    tokenizer,
    begin_tokens: List[str],
    pad_token='<PAD>',
    end_token='<END>',
    validate=False,
):
    answers = [[] for _ in begin_tokens]
    probs = [[] for _ in begin_tokens]
    for seq, score, owner in zip(seqs, scores, belong):
        owner = int(owner.item())
        smi = tokenizer.decode1d(seq.tolist())
        smi = _strip_tokens(
            smi, begin_tokens[owner], end_token=end_token, pad_token=pad_token
        )
        if validate and not check_valid(smi):
            continue
        answers[owner].append(smi)
        probs[owner].append(float(score.item()))
    return answers, probs


@torch.no_grad()
def greedy_inference_batch(
    model,
    tokenizer,
    graphs,
    device,
    begin_tokens,
    max_len,
    end_token='<END>',
    pad_token='<PAD>',
    validate=False,
    use_kv_cache=True,
):
    if isinstance(begin_tokens, str):
        begin_tokens = [begin_tokens]
    start_ids = torch.LongTensor([
        tokenizer.token2idx[x] for x in begin_tokens
    ]).to(device)
    seqs, scores = model.greedy_search(
        graphs=graphs,
        start_ids=start_ids,
        end_idx=tokenizer.token2idx[end_token],
        pad_idx=tokenizer.token2idx[pad_token],
        max_len=max_len,
        left_parenthesis_idx=tokenizer.token2idx['('],
        right_parenthesis_idx=tokenizer.token2idx[')'],
        use_kv_cache=use_kv_cache,
    )
    answers = []
    probs = []
    for idx, seq in enumerate(seqs):
        smi = tokenizer.decode1d(seq.tolist())
        smi = _strip_tokens(
            smi, begin_tokens[idx], end_token=end_token, pad_token=pad_token
        )
        if validate and not check_valid(smi):
            answers.append('')
            probs.append(float('-inf'))
            continue
        answers.append(smi)
        probs.append(float(scores[idx].item()))
    return answers, probs


@torch.no_grad()
def beam_search_batch(
    model,
    tokenizer,
    graphs,
    device,
    begin_tokens,
    max_len,
    size=2,
    pen_para=0,
    end_token='<END>',
    pad_token='<PAD>',
    validate=False,
    use_kv_cache=True,
):
    if isinstance(begin_tokens, str):
        begin_tokens = [begin_tokens]
    start_ids = torch.LongTensor([
        tokenizer.token2idx[x] for x in begin_tokens
    ]).to(device)
    seqs, scores, belong = model.beam_search(
        graphs=graphs,
        start_ids=start_ids,
        end_idx=tokenizer.token2idx[end_token],
        pad_idx=tokenizer.token2idx[pad_token],
        beam=size,
        max_len=max_len,
        left_parenthesis_idx=tokenizer.token2idx['('],
        right_parenthesis_idx=tokenizer.token2idx[')'],
        pen_para=pen_para,
        use_kv_cache=use_kv_cache,
    )
    return _pack_beam_answers(
        seqs=seqs,
        scores=scores,
        belong=belong,
        tokenizer=tokenizer,
        begin_tokens=begin_tokens,
        pad_token=pad_token,
        end_token=end_token,
        validate=validate,
    )


def beam_search_one(
    model,
    tokenizer,
    graph,
    device,
    max_len,
    size=2,
    pen_para=0,
    begin_token='<CLS>',
    end_token='<END>',
    validate=False,
    use_kv_cache=True,
):
    answers, probs = beam_search_batch(
        model=model,
        tokenizer=tokenizer,
        graphs=graph,
        device=device,
        begin_tokens=[begin_token],
        max_len=max_len,
        size=size,
        pen_para=pen_para,
        end_token=end_token,
        validate=validate,
        use_kv_cache=use_kv_cache,
    )
    return answers[0], probs[0]
