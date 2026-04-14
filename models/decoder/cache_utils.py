from typing import List, Optional, Tuple

import torch


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
