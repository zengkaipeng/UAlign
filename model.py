import torch
from sparse_backBone import GATBase
import math


class PositionalEncoding(torch.nn.Module):
    def __init__(self, emb_size: int, dropout: float, maxlen: int = 2000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(
            - torch.arange(0, emb_size, 2) * math.log(10000) / emb_size
        )
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)

        self.dropout = torch.nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: torch.Tensor):
        token_len = token_embedding.shape[1]
        return self.dropout(token_embedding + self.pos_embedding[:token_len])


class PretrainModel(torch.nn.Module):
    def __init__(self, token_size, encoder, decoder, d_model, pos_enc):
        super(PretrainModel, self).__init__()
        self.word_emb = torch.nn.Embedding(token_size, d_model)
        self.encoder, self.decoder = encoder, decoder
        self.pos_enc = pos_enc
        self.output_layer = torch.nn.Sequential(
            torch.nn.Linear(d_model, d_model),
            torch.nn.ReLU(),
            torch.nn.Linear(d_model, token_size)
        )

    def graph2batch(
        self, node_feat: torch.Tensor, batch_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, max_node = batch_mask.shape
        answer = torch.zeros(batch_size, max_node, node_feat.shape[-1])
        answer = answer.to(node_feat)
        answer[batch_mask] = node_feat
        return answer

    def encode(self, graphs):
        node_feat, edge_feat = self.encoder(graphs)
        memory = self.graph2batch(node_feat, graphs.batch_mask)
        memory = self.pos_enc(memory)

        return memory, torch.logical_not(graphs.batch_mask)

    def decode(
        self, tgt, memory, memory_padding_mask=None,
        tgt_mask=None, tgt_padding_mask=None
    ):
        tgt_emb = self.pos_enc(self.word_emb(tgt))
        result = self.decoder(
            tgt=tgt_emb, memory=memory, tgt_mask=tgt_mask,
            memory_key_padding_mask=memory_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask
        )
        return self.output_layer(result)

    def forward(self, graphs, tgt, tgt_mask, tgt_pad_mask):

        memory, memory_pad = self.encode(graphs)
        result = self.decode(
            tgt=tgt, memory=memory, memory_padding_mask=memory_pad,
            tgt_padding_mask=tgt_pad_mask, tgt_mask=tgt_mask
        )

        return result
