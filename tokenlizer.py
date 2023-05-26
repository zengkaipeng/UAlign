import json

DEFAULT_SP = {'<CLS>', '<UNK>', '<PAD>'}


class tokenizer:
    def __init__(self, tokens, sp_token=None):
        super(tokenizer, self).__init__()
        tokens = set(tokens)
        self.token2idx = {v: idx for idx, v in enumerate(tokens)}
        self.sp_token = sp_token
        if self.sp_token is not None:
            for x in sp_token:
                if x not in self.token2idx:
                    self.token2idx[x] = len(self.token2idx)
        self.idx2token = {v: k for k, v in self.token2idx.items()}

    def encode1d(self, seq, unk_token='<UNK>'):
        deft = self.token2idx[unk_token]
        return [self.token2idx.get(x, deft) for x in seq]

    def encode2d(self, batch, unk_token='<UNK>', pad_token='<PAD>'):
        answer, max_len = [], max(len(x) for x in batch)
        pd = self.token2idx[pad_token]
        for x in batch:
            res = [pd] * max_len
            res[:len(x)] = self.encode1d(x, unk_token=unk_token)
            answer.append(res)
        return answer

    def decode1d(self, seq):
        return ''.join(self.idx2token[x] for x in seq)

    def decode2d(self, seq):
        return [self.encode1d(x) for x in seq]

