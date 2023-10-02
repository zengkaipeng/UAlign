import torch
from typing import Any, Dict, List, Tuple, Optional, Union
from collections.abc import Iterable
import torch.nn.functional as F
from GATconv import MyGATConv
from sparse_backBone import SparseEdgeUpdateLayer, SparseBondEncoder
from ogb.graphproppred.mol_encoder import AtomEncoder


class MhAttnBlock(torch.nn.Module):
    def __init__(
        self, Qdim: int, Kdim: int, Vdim: int, Odim: int, heads: int = 1,
        negative_slope: float = 0.2, dropout: float = 0
    ):
        super(MhAttnBlock, self).__init__()
        self.Qdim, self.Kdim, self.Vdim = Qdim, Kdim, Vdim
        self.heads, self.Odim = heads, Odim
        self.negative_slope = negative_slope
        self.LinearK = torch.nn.Linear(Kdim, heads, bias=False)
        self.LinearQ = torch.nn.Linear(Qdim, heads, bias=False)
        self.bias = torch.nn.Parameter(torch.zeros(heads, Odim))
        self.LinearV = torch.nn.Linear(Vdim, heads * Odim, bias=False)
        self.dropout_fun = torch.nn.Dropout(dropout)

        torch.nn.init.xavier_uniform_(self.alphaK)
        torch.nn.init.xavier_uniform_(self.alphaQ)
        torch.nn.init.xavier_uniform_(self.bias)

    def forward(
        self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        (batch_size, Qsize), Ksize = Q.shape[:2], K.shape[1]
        Vproj = self.LinearV(V).reshape(batch_size, -1, self.heads, self.Odim)

        attn_K = self.LinearK(K)  # [B, L, H]
        attn_Q = self.LinearQ(Q)  # [B, L, H]

        attn_K = attn_K.unsqueeze(dim=1).repeat(1, Qsize, 1, 1)
        attn_Q = attn_Q.unsqueeze(dim=2).repeat(1, 1, Ksize, 1)
        attn_w = F.leaky_relu(attn_K + attn_Q, self.negative_slope)

        over_all_mask = self.merge_mask(attn_mask, key_padding_mask)

        if over_all_mask is not None:
            attn_w = torch.masked_fill(attn_w, over_all_mask, 1 - (1 << 32))

        attn_w = self.dropout_fun(torch.softmax(attn_w, dim=2).unsqueeze(-1))
        x_out = (attn_w * Vproj.unsqueeze(dim=1)).sum(dim=2) + self.bias
        return x_out.reshape(batch_size, Qsize, -1)

    def merge_mask(self, attn_mask, key_padding_mask):
        if key_padding_mask is not None:
            batch_size, max_len = key_padding_mask.shape
            mask_shape = (batch_size, max_len, max_len, self.heads)
            all_mask = torch.zeros(mask_shape).to(key_padding_mask)
            all_mask[key_padding_mask] = True
            all_mask = all_mask.transpose(0, 1)
            all_mask[key_padding_mask] = True
            if attn_mask is not None:
                all_mask = torch.logical_and(all_mask, attn_mask)
            return all_mask
        else:
            return attn_mask if attn_mask is not None else None


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
        self, X: torch.Tensor, attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.model(
            Q=X, K=X, V=X, attn_mask=attn_mask,
            key_padding_mask=key_padding_mask
        )


class MixConv(torch.nn.Module):
    def __init__(
        self, in_channels, out_channels, edge_dim, heads=2,
        negative_slope=0.2, dropout=0.1, add_self_loop=True,
    ):
        super(MixConv, self).__init__()
        self.attn = SelfAttnBlock(
            in_channels, out_channels, heads=heads,
            negative_slope=negative_slope, dropout=dropout
        )
        self.conv = MyGATConv(
            in_channels, out_channels, edge_dim, heads=heads,
            negative_slope=negative_slope, dropout=dropout,
            add_self_loop=add_self_loop
        )

    def forward(
        self, x, edge_index, edge_attr, key_padding_mask, attn_mask=None
    ):
        conv_res = self.conv(x, edge_index, edge_attr)
        (batch_size, max_len), dim = key_padding_mask.shape, x.shape[-1]
        all_x = torch.zeros((batch_size, max_len, dim)).to(x)
        all_x[~key_padding_mask] = x
        attn_res = self.attn(
            all_x, attn_mask=attn_mask, key_padding_mask=key_padding_mask
        )

        return torch.cat([conv_res, all_x[~key_padding_mask]], dim=-1)


def MixFormer(torch.nn.Module):
    def __init__(
        self, emb_dim, num_layer, heads=2, dropout=0.1,
        negative_slope=0.2,  add_self_loop=True,
    ):
        super(MixFormer, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        self.edge_update = torch.nn.ModuleList()
        self.num_layers, self.num_heads = num_layers, num_heads
        assert emb_dim % (heads * 2) == 0, \
            'The embedding dim should be evenly divided by num_heads'
        self.drop_fun = torch.nn.Dropout(dropout)
        for x in range(num_layer):
            self.batch_norms.append(torch.nn.LayerNorm(emb_dim))
            self.edge_update.append(SparseEdgeUpdateLayer(
                node_dim=emb_dim, edge_dim=emb_dim
            ))
            self.convs.append(MixConv(
                in_channels=emb_dim, out_channels=emb_dim // (heads * 2),
                edge_dim=emb_dim, heads=heads, negative_slope=negative_slope,
                dropout=dropout, add_self_loop=add_self_loop
            ))

        self.atom_encoder = AtomEncoder(emb_dim)
        self.bond_encoder = SparseBondEncoder(emb_dim)

    def forward(self, graph):
        node_feats = self.atom_encoder(graph.x)
        edge_feats = self.bond_encoder(
            edge_feat=graph.edge_attr, org_mask=graph.org_mask,
            self_mask=graph.self_mask
        )
        for layer in range(self.num_layers):
            conv_res = self.batch_norms[layer](self.convs[layer](
                x=node_feats, edge_attr=edge_feats,
                edge_index=graph.edge_index,
                key_padding_mask=torch.logical_not(graph.batch_mask)
            ))

            node_feats = self.dropout_fun(torch.relu(conv_res)) + node_feats
            edge_feats = self.edge_update[layer](
                edge_feats=edge_feats, node_feats=node_feats,
                edge_index=graph.edge_index
            )

        return node_feats, edge_feats
