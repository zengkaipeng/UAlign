import math
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F


KVItem = Optional[Tuple[torch.Tensor, torch.Tensor]]
KVCache = List[KVItem]


def select_kv_cache(cache: KVCache, index: torch.Tensor) -> KVCache:
    selected = []
    for layer_cache in cache:
        if layer_cache is None:
            selected.append(None)
            continue
        key, value = layer_cache
        selected.append((key[index], value[index]))
    return selected


def repeat_kv_cache(cache: KVCache, repeat: int) -> KVCache:
    if repeat <= 0:
        raise ValueError(f'repeat should be positive, got {repeat}')
    repeated = []
    for layer_cache in cache:
        if layer_cache is None:
            repeated.append(None)
            continue
        key, value = layer_cache
        repeated.append((
            key.repeat_interleave(repeat, dim=0),
            value.repeat_interleave(repeat, dim=0)
        ))
    return repeated


class CachedTransformerDecoderLayer(torch.nn.TransformerDecoderLayer):
    @staticmethod
    def _scaled_dot_product_attention(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        dropout_p: float = 0.0,
        training: bool = False,
    ) -> torch.Tensor:
        if hasattr(F, 'scaled_dot_product_attention'):
            return F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=False,
            )

        scale = 1.0 / math.sqrt(q.shape[-1])
        attn_bias = torch.zeros(
            q.shape[:-1] + (k.shape[-2],),
            dtype=q.dtype,
            device=q.device,
        )
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_bias = attn_bias.masked_fill(
                    attn_mask, float('-inf')
                )
            else:
                attn_bias = attn_bias + attn_mask.to(q.dtype)
        attn_score = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn_score = attn_score + attn_bias
        attn_prob = torch.softmax(attn_score, dim=-1)
        if training and dropout_p > 0:
            attn_prob = F.dropout(attn_prob, p=dropout_p, training=True)
        return torch.matmul(attn_prob, v)

    @staticmethod
    def _split_heads(x: torch.Tensor, num_heads: int) -> torch.Tensor:
        batch_size, seq_len, model_dim = x.shape
        head_dim = model_dim // num_heads
        x = x.reshape(batch_size, seq_len, num_heads, head_dim)
        return x.permute(0, 2, 1, 3).contiguous()

    @staticmethod
    def _merge_heads(x: torch.Tensor) -> torch.Tensor:
        batch_size, num_heads, seq_len, head_dim = x.shape
        x = x.permute(0, 2, 1, 3).contiguous()
        return x.reshape(batch_size, seq_len, num_heads * head_dim)

    @staticmethod
    def _to_batch_first(x: torch.Tensor, batch_first: bool) -> torch.Tensor:
        return x if batch_first else x.transpose(0, 1).contiguous()

    @staticmethod
    def _from_batch_first(x: torch.Tensor, batch_first: bool) -> torch.Tensor:
        return x if batch_first else x.transpose(0, 1).contiguous()

    @staticmethod
    def _project_q_only(
        mha: torch.nn.MultiheadAttention, query: torch.Tensor
    ) -> torch.Tensor:
        embed_dim = mha.embed_dim
        weight_q = mha.in_proj_weight[:embed_dim]
        bias_q = None if mha.in_proj_bias is None else mha.in_proj_bias[:embed_dim]
        projected = F.linear(query, weight_q, bias_q)
        return CachedTransformerDecoderLayer._split_heads(
            projected, mha.num_heads
        )

    @staticmethod
    def _project_kv(
        mha: torch.nn.MultiheadAttention, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        embed_dim = mha.embed_dim
        weight_k = mha.in_proj_weight[embed_dim: 2 * embed_dim]
        weight_v = mha.in_proj_weight[2 * embed_dim:]
        if mha.in_proj_bias is None:
            bias_k = None
            bias_v = None
        else:
            bias_k = mha.in_proj_bias[embed_dim: 2 * embed_dim]
            bias_v = mha.in_proj_bias[2 * embed_dim:]
        key = F.linear(x, weight_k, bias_k)
        value = F.linear(x, weight_v, bias_v)
        return (
            CachedTransformerDecoderLayer._split_heads(key, mha.num_heads),
            CachedTransformerDecoderLayer._split_heads(value, mha.num_heads),
        )

    @staticmethod
    def _static_attention(
        mha: torch.nn.MultiheadAttention,
        query: torch.Tensor,
        static_k: torch.Tensor,
        static_v: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        training: bool = False,
    ) -> torch.Tensor:
        if mha.add_zero_attn or mha.bias_k is not None or mha.bias_v is not None:
            raise NotImplementedError(
                'KV-cache path does not support add_zero_attn/bias_k/bias_v'
            )

        q = CachedTransformerDecoderLayer._project_q_only(mha, query)
        attn_mask = None
        if key_padding_mask is not None:
            batch_size = query.shape[0]
            src_len = static_k.shape[2]
            attn_mask = torch.zeros(
                (batch_size, 1, 1, src_len),
                dtype=q.dtype,
                device=q.device,
            )
            attn_mask = attn_mask.masked_fill(
                key_padding_mask[:, None, None, :], float('-inf')
            )
        output = CachedTransformerDecoderLayer._scaled_dot_product_attention(
            q,
            static_k,
            static_v,
            attn_mask=attn_mask,
            dropout_p=mha.dropout if training else 0.0,
            training=training,
        )
        output = CachedTransformerDecoderLayer._merge_heads(output)
        return mha.out_proj(output)

    def project_memory_kv_cache(
        self, memory: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        memory = self._to_batch_first(memory, self.self_attn.batch_first)
        return self._project_kv(self.multihead_attn, memory)

    def forward_kv_cache(
        self,
        tgt: torch.Tensor,
        self_cache: Optional[Tuple[torch.Tensor, torch.Tensor]],
        memory_k: torch.Tensor,
        memory_v: torch.Tensor,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        batch_first = self.self_attn.batch_first
        x = self._to_batch_first(tgt, batch_first)
        if x.shape[1] != 1:
            raise ValueError(
                f'forward_kv_cache expects one-step input, got {tuple(tgt.shape)}'
            )

        self_attn = self.self_attn
        cross_attn = self.multihead_attn
        if self.norm_first:
            x_norm = self.norm1(x)
            key_new, value_new = self._project_kv(self_attn, x_norm)
            if self_cache is None:
                key_all, value_all = key_new, value_new
            else:
                key_all = torch.cat([self_cache[0], key_new], dim=2)
                value_all = torch.cat([self_cache[1], value_new], dim=2)
            self_out = self._static_attention(
                self_attn, x_norm, key_all, value_all, training=self.training
            )
            x = x + self.dropout1(self_out)

            cross_in = self.norm2(x)
            cross_out = self._static_attention(
                cross_attn,
                cross_in,
                memory_k,
                memory_v,
                key_padding_mask=memory_key_padding_mask,
                training=self.training,
            )
            x = x + self.dropout2(cross_out)
            x = x + self._ff_block(self.norm3(x))
        else:
            key_new, value_new = self._project_kv(self_attn, x)
            if self_cache is None:
                key_all, value_all = key_new, value_new
            else:
                key_all = torch.cat([self_cache[0], key_new], dim=2)
                value_all = torch.cat([self_cache[1], value_new], dim=2)
            self_out = self._static_attention(
                self_attn, x, key_all, value_all, training=self.training
            )
            x = self.norm1(x + self.dropout1(self_out))

            cross_out = self._static_attention(
                cross_attn,
                x,
                memory_k,
                memory_v,
                key_padding_mask=memory_key_padding_mask,
                training=self.training,
            )
            x = self.norm2(x + self.dropout2(cross_out))
            x = self.norm3(x + self._ff_block(x))

        return self._from_batch_first(x, batch_first), (key_all, value_all)


class CachedTransformerDecoder(torch.nn.TransformerDecoder):
    def build_memory_cache(self, memory: torch.Tensor):
        cache = []
        for layer in self.layers:
            if not isinstance(layer, CachedTransformerDecoderLayer):
                raise TypeError(
                    'CachedTransformerDecoder requires '
                    'CachedTransformerDecoderLayer layers'
                )
            cache.append(layer.project_memory_kv_cache(memory))
        return cache

    def forward_kv_cache(
        self,
        tgt: torch.Tensor,
        self_cache,
        memory_cache,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
        memory_select_idx: Optional[torch.Tensor] = None,
    ):
        new_cache = []
        output = tgt
        for idx, layer in enumerate(self.layers):
            memory_k, memory_v = memory_cache[idx]
            if memory_select_idx is not None:
                memory_k = memory_k[memory_select_idx]
                memory_v = memory_v[memory_select_idx]
                if memory_key_padding_mask is not None:
                    memory_pad = memory_key_padding_mask[memory_select_idx]
                else:
                    memory_pad = None
            else:
                memory_pad = memory_key_padding_mask

            output, layer_cache = layer.forward_kv_cache(
                tgt=output,
                self_cache=self_cache[idx],
                memory_k=memory_k,
                memory_v=memory_v,
                memory_key_padding_mask=memory_pad,
            )
            new_cache.append(layer_cache)

        if self.norm is not None:
            output = self.norm(output)
        return output, new_cache


__all__ = [
    'CachedTransformerDecoder',
    'CachedTransformerDecoderLayer',
    'repeat_kv_cache',
    'select_kv_cache',
]
