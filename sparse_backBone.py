import torch
from typing import Any, Dict, List, Tuple, Optional, Union
from ogb.graphproppred.mol_encoder import BondEncoder
from torch_geometric.data import Data
from GATconv import MyGATConv
from GINConv import MyGINConv
import numpy as np


class SparseEdgeUpdateLayer(torch.nn.Module):
    def __init__(
        self,
        edge_dim: int = 64,
        node_dim: int = 64,
    ):
        super(SparseEdgeUpdateLayer, self).__init__()
        input_dim = node_dim * 2 + edge_dim
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(input_dim, input_dim),
            torch.nn.LayerNorm(input_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(input_dim, edge_dim)
        )

    def forward(
        self,
        node_feats: torch.Tensor, edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        row, col = edge_index
        node_i, node_j = node_feats[row], node_feats[col]
        x = torch.cat([node_i, node_j, edge_feats], dim=-1)
        return self.mlp(x) + edge_feats


class GINBase(torch.nn.Module):
    def __init__(
        self,  num_layers: int = 4,
        embedding_dim: int = 64,
        dropout: float = 0.7,
    ):
        super(GINBase, self).__init__()
        if num_layers < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        self.edge_update = torch.nn.ModuleList()
        self.num_layers = num_layers
        self.dropout_fun = torch.nn.Dropout(dropout)
        for layer in range(self.num_layers):
            self.convs.append(MyGINConv(embedding_dim))
            self.batch_norms.append(torch.nn.LayerNorm(embedding_dim))
            self.edge_update.append(SparseEdgeUpdateLayer(
                embedding_dim, embedding_dim
            ))

    def forward(
        self,
        node_feats: torch.Tensor, edge_feats: torch.Tensor,
        edge_index: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        num_nodes = node_feats.shape[0]
        for layer in range(self.num_layers):
            conv_res = self.batch_norms[layer](self.convs[layer](
                node_feats=node_feats, edge_feats=edge_feats,
                edge_index=edge_index, num_nodes=num_nodes
            ))

            node_feats = self.dropout_fun(torch.relu(conv_res)) + node_feats

            edge_feats = self.edge_update[layer](
                edge_feats=edge_feats, node_feats=node_feats,
                edge_index=edge_index
            )
        return node_feats, edge_feats


class GATBase(torch.nn.Module):
    def __init__(
        self, num_layers: int = 4, num_heads: int = 4,
        embedding_dim: int = 64, dropout: float = 0.7,
        negative_slope: float = 0.2, add_self_loop: bool = True
    ):
        super(GATBase, self).__init__()
        if num_layers < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        self.edge_update = torch.nn.ModuleList()
        self.num_layers, self.num_heads = num_layers, num_heads
        self.dropout_fun = torch.nn.Dropout(dropout)
        assert embedding_dim % num_heads == 0, \
            'The embedding dim should be evenly divided by num_heads'
        for layer in range(self.num_layers):
            self.convs.append(MyGATConv(
                in_channels=embedding_dim, heads=num_heads,
                out_channels=embedding_dim // num_heads,
                negative_slope=negative_slope, dropout=dropout,
                edge_dim=embedding_dim, add_self_loop=add_self_loop
            ))
            self.batch_norms.append(torch.nn.LayerNorm(embedding_dim))
            self.edge_update.append(SparseEdgeUpdateLayer(
                embedding_dim, embedding_dim
            ))
        self.add_self_loop = add_self_loop

    def forward(
        self,
        node_feats: torch.Tensor, edge_feats: torch.Tensor,
        edge_index: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        for layer in range(self.num_layers):
            conv_res = self.batch_norms[layer](self.convs[layer](
                x=node_feats, edge_attr=edge_feats, edge_index=edge_index
            ))
            node_feats = self.dropout_fun(torch.relu(conv_res)) + node_feats
            edge_feats = self.edge_update[layer](
                edge_feats=edge_feats, node_feats=node_feats,
                edge_index=edge_index
            )

        return node_feats, edge_feats


class SparseBondEncoder(torch.nn.Module):
    def __init__(self, dim):
        super(SparseBondEncoder, self).__init__()
        self.bond_encoder = BondEncoder(dim)
        self.self_embedding = torch.nn.Parameter(torch.randn(dim))
        self.n_class = n_class
        self.dim = dim

    def forward(self, edge_feat, org_mask, self_mask):
        assert org_mask.shape == self_mask.shape, \
            'org mask and self mask should share the same shape'

        num_edges = org_mask.shape[0]
        result = torch.zeros(num_edges, self.dim).to(edge_feat.device)
        result[org_mask] = self.bond_encoder(edge_feat)
        result[self_mask] = self.self_embedding

        return result