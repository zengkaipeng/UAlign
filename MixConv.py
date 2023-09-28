import torch
from typing import Any, Dict, List, Tuple, Optional, Union
from collections.abc import Iterable
import torch.nn.functional as F
from GATconv import MyGATConv
from GINConv import MyGINConv


class MhAttnBlock(torch.nn.Module):
    def __init__(
        self, Qdim: int, Kdim: int, Vdim: int, Odim: int, heads: int = 1,
        negative_slope: float = 0.2, dropout: float = 0
    ):
        super(MhAttnBlock, self).__init__()
        self.Qdim, self.Kdim, self.Vdim = Qdim, Kdim, Vdim
        self.heads, self.Odim = heads, Odim
        self.negative_slope = negative_slope
        self.LinearK = torch.nn.Linear(Kdim, heads * Odim, bias=False)
        self.LinearQ = torch.nn.Linear(Qdim, heads * Odim, bias=False)
        self.alphaQ = torch.nn.Parameter(torch.zeros(1, 1, heads, Odim))
        self.alphaK = torch.nn.Parameter(torch.zeros(1, 1, heads, Odim))
        self.bias = torch.nn.Parameter(torch.zeros(heads, Odim))
        self.LinearV = torch.nn.Linear(Vdim, heads * Odim, bias=False)
        self.dropout_fun = torch.nn.Dropout(dropout)

        torch.nn.init.xavier_uniform_(self.alphaK)
        torch.nn.init.xavier_uniform_(self.alphaQ)
        torch.nn.init.xavier_uniform_(self.bias)

    def forward(
        self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        (batch_size, Qsize), Ksize = Q.shape[:2], K.shape[1]
        Qproj = self.LinearQ(Q).reshape(batch_size, -1, self.heads, self.Odim)
        Kproj = self.LinearK(K).reshape(batch_size, -1, self.heads, self.Odim)
        Vproj = self.LinearV(V).reshape(batch_size, -1, self.heads, self.Odim)

        attn_Q = (self.alphaQ * Qproj).sum(dim=-1)
        attn_K = (self.alphaK * Kproj).sum(dim=-1)

        attn_K = attn_K.unsqueeze(dim=1).repeat(1, Qsize, 1, 1)
        attn_Q = attn_Q.unsqueeze(dim=2).repeat(1, 1, Ksize, 1)
        attn_w = F.leaky_relu(attn_K + attn_Q, self.negative_slope)

        if attn_mask is not None:
            attn_mask = torch.logical_not(attn_mask.unsqueeze(dim=-1))
            INF = (1 << 32) - 1
            attn_w = torch.masked_fill(attn_w, attn_mask, -INF)
        attn_w = self.dropout_fun(torch.softmax(attn_w, dim=2).unsqueeze(-1))
        x_out = (attn_w * Vproj.unsqueeze(dim=1)).sum(dim=2) + self.bias
        return x_out.reshape(batch_size, Qsize, -1)


class SelfAttnBlock(torch.nn.Module):
    def __init__(
        self, input_dim: int, output_dim: int, heads: int = 1,
        negative_slope: float = 0.2, dropout: float = 0
    ):
        super(SelfAttnBlock, self).__init__()
        self.model = MhAttnBlock(
            Qdim=input_dim, Kdim=input_dim, Vdim=input_dim, heads=heads,
            Odim=output_dim, negative_slope=negative_slope, dropout=dropout
        )

    def forward(
        self, X: torch.Tensor, attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return self.model(Q=X, K=X, V=X, attn_mask=attn_mask)
