import re
import json
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw

DEFAULT_SP = {'<CLS>', '<UNK>', '<PAD>', "<END>"}


SMI_REGEX_PATTERN = r"""(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$|\%[0-9]{2}|[0-9])"""


class Tokenizer:
    def __init__(self, tokens, sp_token=None):
        super(Tokenizer, self).__init__()
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

    def encode2d(
        self, batch, unk_token='<UNK>', pad_token='<PAD>',
        max_len=None
    ):
        if max_len is None:
            max_len = max(len(x) for x in batch)
        pd, answer = self.token2idx[pad_token], []
        for x in batch:
            res, t_len = [pd] * max_len, min(len(x), max_len)
            res[:t_len] = self.encode1d(x[:t_len], unk_token=unk_token)
            answer.append(res)
        return answer

    def decode1d(self, seq):
        return ''.join(self.idx2token[x] for x in seq)

    def get_token_size(self):
        return len(self.token2idx)

    def decode2d(self, seq):
        return [self.decode1d(x) for x in seq]


def smi_tokenizer(smi):
    """
    Tokenize a SMILES molecule or reaction
    """
    regex = re.compile(SMI_REGEX_PATTERN)
    tokens = [token for token in regex.findall(smi)]
    # assert smi == ''.join(tokens), f"smi is {smi}"
    if smi != ''.join(tokens):
        print('[WARNING] Unseen Tokens Found')
        print('[ORG SMILES]', smi)
        print('[NEW SMILES]', ''.join(tokens))
    return tokens


if __name__ == '__main__':
    print(smi_tokenizer('CC(=O)OC%11=CC=CC=C%11C(=O)O'))
    print(smi_tokenizer('O=C1CCC(=O)N1Br.C/C=C/C(=O)O[Si](C)(C)C'))
    print(smi_tokenizer('[OH-]'))
    print(smi_tokenizer('[Ti++++]'))
