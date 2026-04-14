import json
import math
from typing import Optional

import torch

from .decoder import (
    CachedTransformerDecoder,
    CachedTransformerDecoderLayer,
    repeat_kv_cache,
    select_kv_cache,
)
from .sparse_backBone import GATBase


REQUIRED_MODEL_ARCH_KEYS = (
    'dim',
    'n_layer',
    'heads',
    'negative_slope',
    'dropout',
)


def normalize_model_arch(model_arch):
    missing = [
        key for key in REQUIRED_MODEL_ARCH_KEYS if key not in model_arch
    ]
    if missing:
        raise KeyError(
            'Missing keys in model_arch: ' + ', '.join(sorted(missing))
        )
    normalized = dict(model_arch)
    normalized['dim'] = int(normalized['dim'])
    normalized['n_layer'] = int(normalized['n_layer'])
    normalized['heads'] = int(normalized['heads'])
    normalized['negative_slope'] = float(normalized['negative_slope'])
    normalized['dropout'] = float(normalized['dropout'])
    return normalized


def load_model_arch(model_arch_path):
    with open(model_arch_path) as fin:
        return normalize_model_arch(json.load(fin))


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

    @classmethod
    def from_arch(
        cls,
        token_size,
        model_arch,
        use_class=False,
        max_len=2000,
    ):
        model_arch = normalize_model_arch(model_arch)
        d_model = model_arch['dim']
        n_layer = model_arch['n_layer']
        heads = model_arch['heads']
        negative_slope = model_arch['negative_slope']
        dropout = model_arch['dropout']

        encoder = GATBase(
            num_layers=n_layer,
            dropout=dropout,
            embedding_dim=d_model,
            num_heads=heads,
            negative_slope=negative_slope,
            n_class=11 if use_class else None,
        )
        decode_layer = CachedTransformerDecoderLayer(
            d_model=d_model,
            nhead=heads,
            batch_first=True,
            dim_feedforward=d_model * 2,
            dropout=dropout,
        )
        decoder = CachedTransformerDecoder(decode_layer, n_layer)
        pos_enc = PositionalEncoding(d_model, dropout, maxlen=max_len)
        return cls(
            token_size=token_size,
            encoder=encoder,
            decoder=decoder,
            d_model=d_model,
            pos_enc=pos_enc,
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

    def _embed_last_step(self, token_ids: torch.Tensor, step_idx: int):
        step_emb = self.word_emb(token_ids)
        pos_emb = self.pos_enc.pos_embedding[step_idx: step_idx + 1].to(step_emb)
        return self.pos_enc.dropout(step_emb + pos_emb)

    @staticmethod
    def _generate_causal_mask(seq_len: int, device):
        mask = torch.triu(
            torch.ones((seq_len, seq_len), dtype=torch.bool, device=device),
            diagonal=1,
        )
        return mask

    def _build_decoder_cache(self, memory: torch.Tensor):
        if not isinstance(self.decoder, CachedTransformerDecoder):
            raise TypeError(
                'KV-cache decoding requires CachedTransformerDecoder'
            )
        return self.decoder.build_memory_cache(memory)

    def _decode_next_token(
        self,
        seq: torch.Tensor,
        memory: torch.Tensor,
        memory_pad: Optional[torch.Tensor],
        memory_select_idx: torch.Tensor,
        use_kv_cache: bool,
        self_cache=None,
        memory_cache=None,
    ):
        if use_kv_cache:
            step_emb = self._embed_last_step(seq[:, -1:], seq.shape[1] - 1)
            dec_out, new_cache = self.decoder.forward_kv_cache(
                tgt=step_emb,
                self_cache=self_cache,
                memory_cache=memory_cache,
                memory_key_padding_mask=memory_pad,
                memory_select_idx=memory_select_idx,
            )
            return self.output_layer(dec_out)[:, -1], new_cache

        token_logits = self.decode(
            tgt=seq,
            memory=memory[memory_select_idx],
            memory_padding_mask=memory_pad[memory_select_idx]
            if memory_pad is not None else None,
            tgt_mask=self._generate_causal_mask(seq.shape[1], seq.device),
        )[:, -1]
        return token_logits, None

    def greedy_search(
        self,
        graphs,
        start_ids: torch.Tensor,
        end_idx: int,
        pad_idx: int,
        max_len: int = 400,
        left_parenthesis_idx: int = -1,
        right_parenthesis_idx: int = -1,
        use_kv_cache: bool = True,
    ):
        if start_ids.dim() == 1:
            seq = start_ids.unsqueeze(-1)
        elif start_ids.dim() == 2 and start_ids.shape[1] == 1:
            seq = start_ids
        else:
            raise ValueError('start_ids should have shape [bs] or [bs, 1]')

        memory, memory_pad = self.encode(graphs)
        batch_size = seq.shape[0]
        device = seq.device
        seq_scores = torch.zeros(batch_size, device=device)
        alive = torch.ones(batch_size, dtype=torch.bool, device=device)
        balance = torch.zeros(batch_size, dtype=torch.long, device=device)

        if use_kv_cache:
            memory_cache = self._build_decoder_cache(memory)
            alive_cache = [None for _ in self.decoder.layers]
        else:
            memory_cache = None
            alive_cache = None

        for _ in range(max_len):
            if not torch.any(alive).item():
                break

            alive_idx = torch.where(alive)[0]
            token_logits, next_cache = self._decode_next_token(
                seq=seq[alive],
                memory=memory,
                memory_pad=memory_pad,
                memory_select_idx=alive_idx,
                use_kv_cache=use_kv_cache,
                self_cache=alive_cache,
                memory_cache=memory_cache,
            )
            token_logp = torch.log_softmax(token_logits, dim=-1)
            pred = torch.argmax(token_logp, dim=-1)
            pred_score = token_logp[
                torch.arange(pred.shape[0], device=device), pred
            ]

            to_append = torch.full(
                (batch_size,), pad_idx, dtype=torch.long, device=device
            )
            to_append[alive] = pred
            seq = torch.cat([seq, to_append.unsqueeze(-1)], dim=-1)
            seq_scores[alive] = seq_scores[alive] + pred_score

            if left_parenthesis_idx >= 0 and right_parenthesis_idx >= 0:
                balance[alive] = balance[alive] + (
                    (pred == left_parenthesis_idx).long() -
                    (pred == right_parenthesis_idx).long()
                )
                illegal = balance[alive] < 0
                if torch.any(illegal).item():
                    seq_scores[alive_idx[illegal]] = float('-inf')
                unclosed = (pred == end_idx) & (balance[alive] != 0)
                if torch.any(unclosed).item():
                    seq_scores[alive_idx[unclosed]] = float('-inf')

            keep_alive = pred != end_idx
            if left_parenthesis_idx >= 0 and right_parenthesis_idx >= 0:
                keep_alive = keep_alive & (balance[alive] >= 0)
            alive[alive_idx] = keep_alive

            if use_kv_cache:
                if torch.any(keep_alive).item():
                    alive_cache = select_kv_cache(
                        next_cache, torch.where(keep_alive)[0]
                    )
                else:
                    alive_cache = [None for _ in self.decoder.layers]

        return seq, seq_scores

    def beam_search(
        self,
        graphs,
        start_ids: torch.Tensor,
        end_idx: int,
        pad_idx: int,
        beam: int = 10,
        max_len: int = 400,
        left_parenthesis_idx: int = -1,
        right_parenthesis_idx: int = -1,
        pen_para: float = 0,
        use_kv_cache: bool = True,
    ):
        if beam <= 0:
            raise ValueError(f'beam should be positive, got {beam}')
        if start_ids.dim() == 1:
            seq = start_ids.unsqueeze(-1)
        elif start_ids.dim() == 2 and start_ids.shape[1] == 1:
            seq = start_ids
        else:
            raise ValueError('start_ids should have shape [bs] or [bs, 1]')

        if (
            (left_parenthesis_idx < 0) != (right_parenthesis_idx < 0)
        ):
            raise ValueError(
                'left_parenthesis_idx and right_parenthesis_idx '
                'must be provided together'
            )

        memory, memory_pad = self.encode(graphs)
        batch_size = seq.shape[0]
        device = seq.device
        scores = torch.zeros(batch_size, device=device)
        lengths = torch.zeros(batch_size, dtype=torch.long, device=device)
        belong = torch.arange(batch_size, device=device)
        alive = torch.ones(batch_size, dtype=torch.bool, device=device)
        balance = torch.zeros(batch_size, dtype=torch.long, device=device)

        if use_kv_cache:
            memory_cache = self._build_decoder_cache(memory)
            alive_cache = [None for _ in self.decoder.layers]
        else:
            memory_cache = None
            alive_cache = None

        for _ in range(max_len):
            dead = ~alive
            if not torch.any(alive).item():
                break

            seq_cand = []
            score_cand = []
            len_cand = []
            belong_cand = []
            balance_cand = []
            alive_cand = []
            cache_ref = []

            if torch.any(dead).item():
                pad_col = torch.full(
                    (dead.sum().item(), 1), pad_idx,
                    dtype=torch.long, device=device
                )
                seq_cand.append(torch.cat([seq[dead], pad_col], dim=-1))
                score_cand.append(scores[dead])
                len_cand.append(lengths[dead])
                belong_cand.append(belong[dead])
                balance_cand.append(balance[dead])
                alive_cand.append(torch.zeros(dead.sum().item(), dtype=torch.bool, device=device))
                if use_kv_cache:
                    cache_ref.append(torch.full(
                        (dead.sum().item(),), -1,
                        dtype=torch.long, device=device
                    ))

            alive_idx = torch.where(alive)[0]
            seq_alive = seq[alive]
            belong_alive = belong[alive]
            token_logits, next_cache = self._decode_next_token(
                seq=seq_alive,
                memory=memory,
                memory_pad=memory_pad,
                memory_select_idx=belong_alive,
                use_kv_cache=use_kv_cache,
                self_cache=alive_cache,
                memory_cache=memory_cache,
            )
            token_logp = torch.log_softmax(token_logits, dim=-1)
            dup = min(token_logp.shape[-1], beam)
            topk = torch.topk(token_logp, k=dup, dim=-1, largest=True, sorted=True)

            expanded_seq = seq_alive[:, None, :].repeat(1, dup, 1)
            expanded_seq = torch.cat(
                [expanded_seq, topk.indices.unsqueeze(-1)], dim=-1
            )
            expanded_scores = scores[alive][:, None] + topk.values
            expanded_lengths = lengths[alive][:, None].repeat(1, dup) + 1
            expanded_belong = belong_alive[:, None].repeat(1, dup)
            expanded_alive = topk.indices != end_idx
            expanded_balance = balance[alive][:, None].repeat(1, dup)

            if left_parenthesis_idx >= 0:
                expanded_balance = expanded_balance + (
                    (topk.indices == left_parenthesis_idx).long() -
                    (topk.indices == right_parenthesis_idx).long()
                )

            seq_cand.append(expanded_seq.reshape(-1, expanded_seq.shape[-1]))
            score_cand.append(expanded_scores.reshape(-1))
            len_cand.append(expanded_lengths.reshape(-1))
            belong_cand.append(expanded_belong.reshape(-1))
            balance_cand.append(expanded_balance.reshape(-1))
            alive_cand.append(expanded_alive.reshape(-1))

            if use_kv_cache:
                expanded_cache = repeat_kv_cache(next_cache, dup)
                cache_ref.append(torch.arange(
                    expanded_alive.numel(), dtype=torch.long, device=device
                ))

            cand_seq = torch.cat(seq_cand, dim=0)
            cand_scores = torch.cat(score_cand, dim=0)
            cand_lengths = torch.cat(len_cand, dim=0)
            cand_belong = torch.cat(belong_cand, dim=0)
            cand_balance = torch.cat(balance_cand, dim=0)
            cand_alive = torch.cat(alive_cand, dim=0)
            if use_kv_cache:
                cand_cache_ref = torch.cat(cache_ref, dim=0)

            illegal = (cand_balance < 0) | ((~cand_alive) & (cand_balance != 0))
            cand_scores[illegal] = float('-inf')
            if 0 < pen_para < 1:
                cand_scores = cand_scores / (
                    cand_lengths.clamp_min(1).float() ** pen_para
                )

            top_seq = []
            top_scores = []
            top_lengths = []
            top_belong = []
            top_balance = []
            top_alive = []
            top_cache_ref = []

            for batch_idx in range(batch_size):
                mask = cand_belong == batch_idx
                if not torch.any(mask).item():
                    continue
                batch_scores = cand_scores[mask]
                keep = min(beam, batch_scores.shape[0])
                order = torch.topk(
                    batch_scores, k=keep, largest=True, sorted=True
                ).indices
                top_seq.append(cand_seq[mask][order])
                top_scores.append(batch_scores[order])
                top_lengths.append(cand_lengths[mask][order])
                top_belong.append(cand_belong[mask][order])
                top_balance.append(cand_balance[mask][order])
                top_alive.append(cand_alive[mask][order])
                if use_kv_cache:
                    top_cache_ref.append(cand_cache_ref[mask][order])

            seq = torch.cat(top_seq, dim=0)
            scores = torch.cat(top_scores, dim=0)
            lengths = torch.cat(top_lengths, dim=0)
            belong = torch.cat(top_belong, dim=0)
            balance = torch.cat(top_balance, dim=0)
            alive = torch.cat(top_alive, dim=0)

            if use_kv_cache:
                cache_ref = torch.cat(top_cache_ref, dim=0)
                if torch.any(alive).item():
                    alive_cache = select_kv_cache(
                        expanded_cache, cache_ref[alive]
                    )
                else:
                    alive_cache = [None for _ in self.decoder.layers]

        return seq, scores, belong

    def forward(self, graphs, tgt, tgt_mask, tgt_pad_mask):

        memory, memory_pad = self.encode(graphs)
        result = self.decode(
            tgt=tgt, memory=memory, memory_padding_mask=memory_pad,
            tgt_padding_mask=tgt_pad_mask, tgt_mask=tgt_mask
        )

        return result

