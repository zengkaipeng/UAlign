import torch
from typing import Any, Dict, List, Tuple, Optional, Union
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from GATconv import SelfLoopGATConv as MyGATConv
from GINConv import MyGINConv
import numpy as np


class SparseEdgeUpdateLayer(torch.nn.Module):
    def __init__(
        self,
        edge_dim: int = 64,
        node_dim: int = 64,
        residual: bool = False
    ):
        super(SparseEdgeUpdateLayer, self).__init__()
        input_dim = node_dim * 2 + edge_dim
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(input_dim, input_dim),
            torch.nn.LayerNorm(input_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(input_dim, edge_dim)
        )
        self.residual = residual

    def forward(
        self,
        node_feats: torch.Tensor, edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        node_i = torch.index_select(
            input=node_feats, dim=0, index=edge_index[0]
        )
        node_j = torch.index_select(
            input=node_feats, dim=0, index=edge_index[1]
        )

        x = torch.cat([node_i, node_j, edge_feats], dim=-1)
        return self.mlp(x) + edge_feats if self.residual else self.mlp(x)


class GINBase(torch.nn.Module):
    def __init__(
        self,  num_layers: int = 4,
        embedding_dim: int = 64,
        dropout: float = 0.7,
        n_class: Optional[int] = None
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
            self.convs.append(MyGINConv(
                in_channels=embedding_dim, out_channels=embedding_dim,
                edge_dim=embedding_dim
            ))
            self.batch_norms.append(torch.nn.LayerNorm(embedding_dim))
            self.edge_update.append(SparseEdgeUpdateLayer(
                embedding_dim, embedding_dim, residual=True
            ))
        self.atom_encoder = SparseAtomEncoder(embedding_dim, n_class)
        self.bond_encoder = SparseBondEncoder(embedding_dim, n_class)

    def forward(self, G) -> torch.Tensor:
        node_feats = self.atom_encoder(G.x, G.get('node_rxn', None))
        edge_feats = self.bond_encoder(G.edge_attr, G.get('edge_rxn', None))

        for layer in range(self.num_layers):
            conv_res = self.batch_norms[layer](self.convs[layer](
                x=node_feats, edge_attr=edge_feats, edge_index=G.edge_index,
                org_mask=G.get('e_org_mask', None)
            ))
            node_feats = self.dropout_fun(torch.relu(conv_res)) + node_feats

            if G.get('e_org_mask', None) is not None:
                useful_edges = G.edge_index[:, G.e_org_mask]
            else:
                useful_edges = G.edge_index

            edge_feats = self.edge_update[layer](
                edge_feats=edge_feats, node_feats=node_feats,
                edge_index=useful_edges
            )
        return node_feats, edge_feats


class GATBase(torch.nn.Module):
    def __init__(
        self, num_layers: int = 4, num_heads: int = 4, embedding_dim: int = 64,
        dropout: float = 0.7, negative_slope: float = 0.2,
        n_class: Optional[int] = None
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
                negative_slope=negative_slope,
                dropout=dropout, edge_dim=embedding_dim
            ))
            self.batch_norms.append(torch.nn.LayerNorm(embedding_dim))
            self.edge_update.append(SparseEdgeUpdateLayer(
                embedding_dim, embedding_dim, residual=True
            ))
        self.atom_encoder = SparseAtomEncoder(embedding_dim, n_class)
        self.bond_encoder = SparseBondEncoder(embedding_dim, n_class)

    def forward(self, G) -> torch.Tensor:
        node_feats = self.atom_encoder(G.x, G.get('node_rxn', None))
        edge_feats = self.bond_encoder(G.edge_attr, G.get('edge_rxn', None))
        for layer in range(self.num_layers):
            conv_res = self.batch_norms[layer](self.convs[layer](
                x=node_feats, edge_attr=edge_feats, edge_index=G.edge_index,
                org_mask=G.get('e_org_mask', None)
            ))
            node_feats = self.dropout_fun(torch.relu(conv_res)) + node_feats

            if G.get('e_org_mask', None) is not None:
                useful_edges = G.edge_index[:, G.e_org_mask]
            else:
                useful_edges = G.edge_index

            edge_feats = self.edge_update[layer](
                edge_feats=edge_feats, node_feats=node_feats,
                edge_index=useful_edges
            )

        return node_feats, edge_feats


class SparseAtomEncoder(torch.nn.Module):
    def __init__(self, dim, n_class=None):
        super(SparseAtomEncoder, self).__init__()
        self.atom_encoder = AtomEncoder(dim)
        self.n_class = n_class
        if n_class is not None:
            self.rxn_class_emb = torch.nn.Embedding(n_class, dim)
            self.lin = torch.nn.Linear(dim + dim, dim)
        self.dim = dim

    def forward(self, node_feat, rxn_class=None):
        result = self.atom_encoder(node_feat)
        if self.n_class is not None:
            if rxn_class is None:
                raise ValueError('missing reaction class information')
            else:
                rxn_class_emb = self.rxn_class_emb(rxn_class)
                result = torch.cat([rxn_class_emb, result], dim=-1)
                result = self.lin(result)
        return result


class SparseBondEncoder(torch.nn.Module):
    def __init__(self, dim, n_class=None):
        super(SparseBondEncoder, self).__init__()
        self.bond_encoder = BondEncoder(dim)
        self.n_class = n_class
        if n_class is not None:
            self.rxn_class_emb = torch.nn.Embedding(n_class, dim)
            self.lin = torch.nn.Linear(dim + dim, dim)
        self.dim = dim

    def forward(self, edge_feat, rxn_class=None):
        result = self.bond_encoder(edge_feat)
        if self.n_class is not None:
            if rxn_class is None:
                raise ValueError('missing reaction class information')
            else:
                rxn_class_emb = self.rxn_class_emb(rxn_class)
                result = torch.cat([rxn_class_emb, result], dim=-1)
                result = self.lin(result)
        return result
