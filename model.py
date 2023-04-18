import torch
import torch_scatter
from ogb.utils import smiles2graph
from typing import Any, Dict, List, Tuple, Optional, Union
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder


class EditDataset(torch.utils.data.Dataset):
    def __init__(
        self, graphs: List[Dict],
        activate_nodes: List[List],
        edge_types: List[List[List]],
        edge_edits: List[List[Tuple]]
    ):
        super(EditDataset, self).__init__()
        self.graphs = graphs
        self.activate_nodes = activate_nodes
        self.edge_types = edge_types
        self.edge_edits = edge_edits

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, index):
        node_label = torch.zeros(self.graphs[index]['num_nodes'])
        node_label[self.activate_nodes[index]] = 1
        return self.graphs[index], node_label, \
            self.edge_edits[index], self.edge_types[index]


class ExtendedBondEncoder(torch.nn.Module):
    def __init__(self, dim: int):
        super(ExtendedBondEncoder, self).__init__()
        self.padding_emb = torch.nn.Parameter(torch.zeros(dim))
        self.bond_encoder = BondEncoder(dim)
        self.dim = dim
        torch.nn.init.xavier_uniform_(self.padding_emb)

    def forward(
        self, edge_index: torch.Tensor,
        edge_feat: torch.Tensor, num_nodes: int
    ) -> torch.Tensor:
        edge_result = self.padding_emb.repeat(num_nodes, num_nodes, 1)
        xindex, yindex = edge_index
        edge_result[xindex, yindex] = self.bond_encoder(edge_feat)
        return edge_result


class EdgeUpdateLayer(torch.nn.Module):
    def __init__(
        self,
        edge_dim: int = 64,
        node_dim: int = 64,
        residual: bool = False
    ):
        super(EdgeUpdateLayer, self).__init__()
        input_dim = node_dim * 2 + edge_dim
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(input_dim, input_dim),
            torch.nn.BatchNorm1d(input_dim),
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


def edit_collect_fn(data_batch):
    batch_size = len(data_batch)
    max_node_size = max([x[0]['num_nodes'] for x in data_batch])
    