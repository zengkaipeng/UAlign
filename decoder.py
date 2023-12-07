import torch
from sparse_backBone import (
    SparseAtomEncoder, SparseBondEncoder, SparseEdgeUpdateLayer
)

from GINConv import MyGINConv
from GATconv import SelfLoopGATConv as MyGATConv
from Mix_backbone import MixConv
from typing import Any, Dict, List, Tuple, Optional, Union

from torch.nn import MultiheadAttention
import math


def graph2batch(
    node_feat: torch.Tensor, batch_mask: torch.Tensor,
    batch_size: int, max_node: int
) -> torch.Tensor:
    answer = torch.zeros(batch_size, max_node, node_feat.shape[-1])
    answer = answer.to(node_feat.device)
    answer[batch_mask] = node_feat
    return answer


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


class Feat_init(torch.nn.Module):
    def __init__(
        self, n_pad: int, dim: int, heads: int = 2, dropout: float = 0.1,
        n_class: Optional[int] = None, with_PE: bool = False
    ):
        super(Feat_init, self).__init__()
        self.Qemb = torch.nn.Parameter(torch.randn(1, n_pad, dim))
        self.atom_encoder = SparseAtomEncoder(dim, n_class=None)
        self.bond_encoder = SparseBondEncoder(dim, n_class=None)
        self.with_PE = with_PE
        if with_PE:
            self.pos_enc = PositionalEncoding(dim, dropout)

        if n_class is not None:
            self.node_cls_emb = torch.nn.Embedding(n_class, dim)
            self.edge_cls_emb = torch.nn.Embedding(n_class, dim)
            self.node_lin = torch.nn.Linear(dim << 1, dim)
            self.edge_lin = torch.nn.Linear(dim << 1, dim)

        assert dim % heads == 0, 'dim should be evenly divided by heads'

        self.Attn = MultiheadAttention(
            dim, num_heads=heads, dropout=dropout,
            batch_first=True
        )

        self.feat_ext = torch.nn.Linear(dim << 1, dim)

        self.dim, self.n_class, self.n_pad = dim, n_class, n_pad

    def forward(self, G, memory, mem_pad_mask=None):
        device, batch_size = G.x.device, G.batch.max().item() + 1

        # get node feat
        node_feat = torch.zeros((G.num_nodes, self.dim)).to(device)
        org_node_feat = self.atom_encoder(G.x, None)
        node_feat[G.n_org_mask] = org_node_feat
        Qval = self.Qemb.repeat(batch_size, 1, 1)
        pad_node_feat, _ = self.Attn(
            query=Qval, key=memory, value=memory,
            key_padding_mask=mem_pad_mask
        )

        if self.with_PE:
            pad_node_feat = self.pos_enc(pad_node_feat)

        # [B, pad, dim]
        node_feat[G.n_pad_mask] = pad_node_feat.reshape(-1, self.dim)

        # edge_feat
        num_edges = G.e_org_mask.shape[0]
        edge_feat = torch.zeros(num_edges, self.dim).to(device)
        edge_feat[G.e_org_mask] = self.bond_encoder(G.edge_attr, None)
        row, col = G.edge_index[:, G.e_pad_mask]
        edge_feat[G.e_pad_mask] = self.feat_ext(torch.relu(
            torch.cat([node_feat[row], node_feat[col]], dim=-1)
        ))

        if self.n_class is not None:
            n_cls_emb = self.node_cls_emb(graph.node_rxn)
            node_feat = torch.cat([node_feat, n_cls_emb], dim=-1)
            node_feat = self.node_lin(node_feat)

            e_cls_emb = self.edge_cls_emb(graph.edge_rxn)
            edge_feat = torch.cat([edge_feat, e_cls_emb], dim=-1)
            edge_feat = self.edge_lin(edge_feat)

        return node_feat, edge_feat


class MixDecoder(torch.nn.Module):
    def __init__(
        self, emb_dim: int, n_layers: int, gnn_args: Union[Dict, List[Dict]],
        n_pad: int, dropout: float = 0, heads: int = 1, gnn_type: str = 'gin',
        n_class: Optional[int] = None, update_gate: str = 'add',
        with_PE: bool = False
    ):
        super(MixDecoder, self).__init__()

        self.feat_init = Feat_init(
            n_pad, emb_dim, heads=heads, dropout=dropout,
            n_class=n_class, with_PE=with_PE
        )

        self.num_layers = n_layers
        self.lns = torch.nn.ModuleList()
        self.ln2 = torch.nn.ModuleList()
        self.convs = torch.nn.ModuleList()
        self.edge_update = torch.nn.ModuleList()
        self.cross_attns = torch.nn.ModuleList()

        self.dropout_fun = torch.nn.Dropout(dropout)

        for i in range(self.num_layers):
            self.lns.append(torch.nn.LayerNorm(emb_dim))
            self.ln2.append(torch.nn.LayerNorm(emb_dim))
            gnn_layer = gnn_args[i] if isinstance(gnn_args, list) else gnn_args
            self.convs.append(MixConv(
                emb_dim=emb_dim, gnn_args=gnn_layer, heads=heads,
                dropout=dropout, gnn_type=gnn_type, update_gate=update_gate
            ))
            self.edge_update.append(SparseEdgeUpdateLayer(
                emb_dim, emb_dim, residual=True
            ))
            self.cross_attns.append(MultiheadAttention(
                emb_dim, num_heads=heads, batch_first=True, dropout=dropout
            ))

    def forward(self, graph, memory, mem_pad_mask=None):
        node_feats, edge_feats = self.feat_init(graph, memory, mem_pad_mask)

        batch_size, max_node = graph.batch_mask.shape

        for i in range(self.num_layers):
            conv_res = self.convs[i](
                node_feat=node_feats, edge_feat=edge_feats,
                edge_index=graph.edge_index, batch_mask=graph.batch_mask,
                attn_mask=graph.get('attn_mask', None),
            ) + node_feats

            node_feats = self.dropout_fun(torch.relu(self.lns[i](conv_res)))

            node_feats = graph2batch(
                node_feats, graph.batch_mask, batch_size, max_node
            )

            cross_res = self.cross_attns[i](
                query=node_feats, key=memory, value=memory,
                key_padding_mask=mem_pad_mask,
            )
            cross_res = cross_res[0] + node_feats

            node_feats = torch.relu(self.ln2[i](cross_res))[graph.batch_mask]

            edge_feats = torch.relu(self.edge_update[i](
                edge_feats=edge_feats, node_feats=node_feats,
                edge_index=graph.edge_index
            ))

        return node_feats, edge_feats


class GATDecoder(torch.nn.Module):
    def __init__(
        self, num_layers: int = 4, num_heads: int = 4, embedding_dim: int = 64,
        dropout: float = 0.7, negative_slope: float = 0.2,
        n_class: Optional[int] = None, with_PE: bool = False
    ):
        super(GATDecoder, self).__init__()
        if num_layers < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")
        self.convs = torch.nn.ModuleList()
        self.ln1 = torch.nn.ModuleList()
        self.ln2 = torch.nn.ModuleList()
        self.cross_attns = torch.nn.ModuleList()
        self.edge_update = torch.nn.ModuleList()
        self.num_layers, self.num_heads = num_layers, num_heads
        self.dropout_fun = torch.nn.Dropout(dropout)
        assert embedding_dim % num_heads == 0, \
            'The embedding dim should be evenly divided by num_heads'
        for layer in range(self.num_layers):
            self.convs.append(MyGATConv(
                in_channels=embedding_dim, heads=num_heads,
                out_channels=embedding_dim // num_heads,
                negative_slope=negative_slope,
                dropout=dropout, edge_dim=embedding_dim
            ))
            self.ln1.append(torch.nn.LayerNorm(embedding_dim))
            self.ln2.append(torch.nn.LayerNorm(embedding_dim))
            self.edge_update.append(SparseEdgeUpdateLayer(
                embedding_dim, embedding_dim, residual=True
            ))
            self.cross_attns.append(MultiheadAttention(
                embedding_dim, num_heads=heads,
                batch_first=True, dropout=dropout
            ))

        self.feat_init = Feat_init(
            n_pad, emb_dim, heads=heads, dropout=dropout,
            n_class=n_class, with_PE=with_PE
        )

    def forward(self, graph, memory, mem_pad_mask=None) -> torch.Tensor:
        node_feats, edge_feats = self.feat_init(graph, memory, mem_pad_mask)
        batch_size, max_node = graph.batch_mask.shape

        for layer in range(self.num_layers):
            conv_res = self.ln1[layer](self.convs[layer](
                x=node_feats, edge_attr=edge_feats,
                edge_index=graph.edge_index,
            ))
            node_feats = self.dropout_fun(torch.relu(conv_res)) + node_feats

            node_feats = graph2batch(
                node_feats, graph.batch_mask, batch_size, max_node
            )

            cross_res = self.cross_attns[i](
                query=node_feats, key=memory, value=memory,
                key_padding_mask=mem_pad_mask,
            )
            cross_res = cross_res[0] + node_feats

            node_feats = torch.relu(self.ln2[layer](cross_res))
            node_feats = node_feats[graph.batch_mask]

            edge_feats = torch.relu(self.edge_update[layer](
                edge_feats=edge_feats, node_feats=node_feats,
                edge_index=graph.edge_index
            ))

        return node_feats, edge_feats


class GINDecoder(torch.nn.Module):
    def __init__(
        self,  num_layers: int = 4,
        embedding_dim: int = 64,
        dropout: float = 0.7,
        n_class: Optional[int] = None,
        with_PE: bool = False
    ):
        super(GINDecoder, self).__init__()
        if num_layers < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")
        self.convs = torch.nn.ModuleList()
        self.ln1 = torch.nn.ModuleList()
        self.ln2 = torch.nn.ModuleList()
        self.cross_attns = torch.nn.ModuleList()
        self.edge_update = torch.nn.ModuleList()
        self.num_layers = num_layers
        self.dropout_fun = torch.nn.Dropout(dropout)
        for layer in range(self.num_layers):
            self.convs.append(MyGINConv(
                in_channels=embedding_dim, out_channels=embedding_dim,
                edge_dim=embedding_dim
            ))
            self.ln1.append(torch.nn.LayerNorm(embedding_dim))
            self.ln2.append(torch.nn.LayerNorm(embedding_dim))
            self.edge_update.append(SparseEdgeUpdateLayer(
                embedding_dim, embedding_dim, residual=True
            ))
            self.cross_attns.append(MultiheadAttention(
                embedding_dim, num_heads=heads,
                batch_first=True, dropout=dropout
            ))
        self.feat_init = Feat_init(
            n_pad, emb_dim, heads=heads, dropout=dropout,
            n_class=n_class, with_PE=with_PE
        )

    def forward(self, graph, memory, mem_pad_mask=None) -> torch.Tensor:
        node_feats, edge_feats = self.feat_init(graph, memory, mem_pad_mask)
        batch_size, max_node = graph.batch_mask.shape

        for layer in range(self.num_layers):
            conv_res = self.ln1[layer](self.convs[layer](
                x=node_feats, edge_attr=edge_feats,
                edge_index=graph.edge_index,
            ))
            node_feats = self.dropout_fun(torch.relu(conv_res)) + node_feats
            node_feats = graph2batch(
                node_feats, graph.batch_mask, batch_size, max_node
            )
            cross_res = self.cross_attns[i](
                query=node_feats, key=memory, value=memory,
                key_padding_mask=mem_pad_mask,
            )
            cross_res = cross_res[0] + node_feats

            node_feats = torch.relu(self.ln2[layer](cross_res))
            node_feats = node_feats[graph.batch_mask]

            edge_feats = torch.relu(self.edge_update[layer](
                edge_feats=edge_feats, node_feats=node_feats,
                edge_index=graph.edge_index
            ))
        return node_feats, edge_feats
