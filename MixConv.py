import torch
from typing import Any, Dict, List, Tuple, Optional, Union
from collections.abc import Iterable
import torch.nn.functional as F







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


class MixConv(torch.nn.Module):
    def __init__(
        self, input_dim: int, output_dim: int, heads: int = 1,
        negative_slope: float = 0.2, residual: bool = True,
        dropout: float = 0, attn_dim: Optional[int] = None
    ):
        super(MixConv, self).__init__()
        attn_dim = input_dim if attn_dim is None else attn_dim
        assert attn_dim % heads == 0, 'The dim of attention input' +\
            ' should be evenly divided by num of heads'
        self.attn_conv = SelfAttnBlock(
            input_dim=attn_dim, output_dim=attn_dim // heads,
            heads=heads, negative_slope=negative_slope, dropout=dropout
        )
        self.gin_conv = MyGINConv(input_dim)
        self.ff_block = torch.nn.Sequential(
            torch.nn.Linear(input_dim + attn_dim, output_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(output_dim, output_dim),
        )
        self.ln1 = torch.nn.LayerNorm(input_dim)
        self.ln2 = torch.nn.LayerNorm(input_dim)
        self.residual = residual

    def forward(
        self, node_feat: torch.Tensor, attn_mask: torch.Tensor,
        edge_index: torch.Tensor, edge_feat: torch.Tensor,
        num_nodes: torch.Tensor, ptr: torch.Tensor,
        attn_input: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size, max_node = attn_mask.shape[:2]
        batch_mask = self.batch_mask(ptr, max_node, batch_size)
        attn_input = self.graph2batch(
            attn_input if attn_input is not None else node_feat,
            batch_mask=batch_mask, batch_size=batch_size, max_node=max_node
        )
        attn_res = self.attn_conv(attn_input, attn_mask=attn_mask)
        gin_res = self.gin_conv(node_feat, edge_feat, edge_index, num_nodes)
        if self.residual:
            attn_res = attn_res + attn_input
            gin_res = gin_res + node_feat

        attn_res = self.ln1(attn_res)[batch_mask]
        gin_res = self.ln2(gin_res)

        return self.ff_block(torch.cat([gin_res, attn_res], dim=-1))

    def batch_mask(
        self, ptr: torch.Tensor, max_node: int, batch_size: int
    ) -> torch.Tensor:
        num_nodes = ptr[1:] - ptr[:-1]
        mask = torch.arange(max_node).repeat(batch_size, 1)
        mask = mask.to(num_nodes.device)
        return mask < num_nodes.reshape(-1, 1)

    def graph2batch(
        self, node_feat: torch.Tensor, batch_mask: torch.Tensor,
        batch_size: int, max_node: int
    ) -> torch.Tensor:
        answer = torch.zeros(batch_size, max_node, node_feat.shape[-1])
        answer = answer.to(node_feat.device)
        answer[batch_mask] = node_feat
        return answer