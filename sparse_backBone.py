import torch
import torch_scatter
from typing import Any, Dict, List, Tuple, Optional, Union
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from torch_geometric.data import Data
from GATconv import MyGATConv
import numpy as np


class GINConv(torch.nn.Module):
    def __init__(self, embedding_dim: int = 64):
        super(GINConv, self).__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim, 2 * embedding_dim),
            torch.nn.LayerNorm(2 * embedding_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(2 * embedding_dim, embedding_dim)
        )
        self.eps = torch.nn.Parameter(torch.Tensor([0]))

    def forward(
        self,
        node_feats: torch.Tensor, edge_feats: torch.Tensor,
        edge_index: torch.Tensor, num_nodes: int
    ) -> torch.Tensor:
        message_node = torch.index_select(
            input=node_feats, dim=0, index=edge_index[1]
        )
        message = torch.relu(message_node + edge_feats)
        message_reduce = torch_scatter.scatter(
            message, index=edge_index[0], dim=0,
            dim_size=num_nodes, reduce='sum'
        )

        return self.mlp((1 + self.eps) * node_feats + message_reduce)


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
        node_feat_sum = node_i + node_j
        node_feat_diff = torch.abs(node_i - node_j)

        x = torch.cat([node_feat_sum, node_feat_diff, edge_feats], dim=-1)
        return self.mlp(x) + edge_feats if self.residual else self.mlp(x)


class GINBase(torch.nn.Module):
    def __init__(
        self,  num_layers: int = 4,
        embedding_dim: int = 64,
        dropout: float = 0.7,
        residual: bool = True,
        edge_last: bool = True,
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
            self.convs.append(GINConv(embedding_dim))
            self.batch_norms.append(torch.nn.LayerNorm(embedding_dim))
            if edge_last or layer < self.num_layers - 1:
                self.edge_update.append(SparseEdgeUpdateLayer(
                    embedding_dim, embedding_dim, residual=residual
                ))
        self.residual = residual
        self.edge_last = edge_last

    def forward(
        self,
        node_feats: torch.Tensor, edge_feats: torch.Tensor,
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        num_nodes = node_feats.shape[0]
        for layer in range(self.num_layers):
            node_feats = self.batch_norms[layer](self.convs[layer](
                node_feats=node_feats, edge_feats=edge_feats,
                edge_index=edge_index, 
            ))
            if self.edge_last or layer < self.num_layers - 1:
                edge_feats = self.edge_update[layer](
                    edge_feats=edge_feats, node_feats=node_feats,
                    edge_index=edge_index
                )

            node_feats = self.dropout_fun(
                node_feats if layer == self.num_layers - 1
                else torch.relu(node_feats)
            )
        return node_feats, edge_feats


class GATBase(torch.nn.Module):
    def __init__(
        self, num_layers: int = 4, num_heads: int = 4,
        embedding_dim: int = 64, dropout: float = 0.7,
        residual: bool = True, negative_slope: float = 0.2,
        edge_last: bool = True
    ):
        super(GATBase, self).__init__()
        if num_layers < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        self.edge_update = torch.nn.ModuleList()
        self.num_layers, self.num_heads = num_layers, num_heads
        self.dropout_fun = torch.nn.Dropout(dropout)
        for layer in range(self.num_layers):
            self.convs.append(MyGATConv(
                in_channels=embedding_dim, out_channels=embedding_dim,
                heads=num_heads, negative_slope=negative_slope, 
                dropout=dropout, edge_dim=embedding_dim
            ))
            self.batch_norms.append(torch.nn.LayerNorm(embedding_dim))
            if edge_last or layer < self.num_layers - 1:
                self.edge_update.append(SparseEdgeUpdateLayer(
                    embedding_dim, embedding_dim, residual=residual
                ))
        self.residual = residual
        self.edge_last = edge_last

    def forward(
        self,
        node_feats: torch.Tensor, edge_feats: torch.Tensor,
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        num_nodes = node_feats.shape[0]
        for layer in range(self.num_layers):
            node_feats = self.batch_norms[layer](self.convs[layer](
                x=node_feats, edge_attr=edge_feats,
                edge_index=edge_index, size=num_nodes
            ))
            if self.edge_last or layer < self.num_layers - 1:
                edge_feats = self.edge_update[layer](
                    edge_feats=edge_feats, node_feats=node_feats,
                    edge_index=edge_index
                )

            node_feats = self.dropout_fun(
                node_feats if layer == self.num_layers - 1
                else torch.relu(node_feats)
            )
        return node_feats, edge_feats


def sparse_edit_collect_fn(data_batch):
    batch_size, rxn_class, node_label, num_l = len(data_batch), [], [], []
    edge_idxes, edge_feats, node_feats, lstnode, batch = [], [], [], 0, []
    activate_nodes, num_e, edge_types = [], [], []
    for idx, data in enumerate(data_batch):
        if len(data) == 4:
            graph, n_lb,  e_type, A_node = data
        else:
            graph, r_class, n_lb,  e_type, A_node = data
            rxn_class.append(r_class)

        edge_types.append(e_type)
        node_label.append(n_lb)
        num_l.append(graph['num_nodes'])
        num_e.append(graph['edge_index'].shape[1])

        edge_idxes.append(graph['edge_index'] + lstnode)
        edge_feats.append(graph['edge_feat'])
        node_feats.append(graph['node_feat'])
        lstnode += graph['num_nodes']
        batch.append(np.ones(graph['num_nodes'], dtype=np.int64) * idx)
        activate_nodes.append(A_node)

    result = {
        'edge_index': np.concatenate(edge_idxes, axis=-1),
        'edge_attr': np.concatenate(edge_feats, axis=0),
        'batch': np.concatenate(batch, axis=0),
        'x': np.concatenate(node_feats, axis=0)
    }
    result = {k: torch.from_numpy(v) for k, v in result.items()}
    result['num_nodes'] = lstnode
    node_label = torch.cat(node_label, dim=0)

    if len(rxn_class) == 0:
        return Data(**result), node_label, num_l, num_e,\
            edge_types, activate_nodes
    else:
        return Data(**result), torch.LongTensor(rxn_class),\
            node_label, num_l, num_e, edge_types, activate_nodes


class SparseAtomEncoder(torch.nn.Module):
    def __init__(self, dim, n_class=None):
        super(SparseAtomEncoder, self).__init__()
        self.atom_encoder = AtomEncoder(dim)
        self.n_class = n_class
        if n_class is not None:
            self.rxn_class_emb = torch.nn.Embedding(n_class, dim)
            self.lin = torch.nn.Linear(dim + dim, dim)

    def forward(self, node_feat, num_nodes=None, rxn_class=None):
        result = self.atom_encoder(node_feat)
        if self.n_class is not None:
            if rxn_class is None or num_nodes is None:
                raise ValueError('missing reaction class information')
            else:
                rxn_cls = torch.LongTensor(node_feat.shape[0])
                base, rxn_cls = 0, rxn_cls.to(node_feat.device)
                for idx, p in enumerate(rxn_class):
                    rxn_cls[base: base + num_nodes[idx]] = p
                    base += num_nodes[idx]
                rxn_class_emb = self.rxn_class_emb(rxn_cls)
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

    def forward(self, edge_feat, num_edges=None, rxn_class=None):
        result = self.bond_encoder(edge_feat)
        if self.n_class is not None:
            if rxn_class is None or num_edges is None:
                raise ValueError('missing reaction class information')
            else:
                rxn_cls = torch.LongTensor(node_feat.shape[0])
                base, rxn_cls = 0, rxn_cls.to(node_feat.device)
                for idx, p in enumerate(rxn_class):
                    rxn_cls[base: base + num_edges[idx]] = p
                    base += num_edges
                rxn_class_emb = self.rxn_class_emb(rxn_cls)
                result = torch.cat([rxn_class_emb, result], dim=-1)
                result = self.lin(result)
        return result
