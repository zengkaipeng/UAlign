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
        edge_edits: List[List[Tuple]],
        rxn_class: Optional[List[int]] = None
    ):
        super(EditDataset, self).__init__()
        self.graphs = graphs
        self.activate_nodes = activate_nodes
        self.edge_types = edge_types
        self.edge_edits = edge_edits
        self.rxn_class = rxn_class

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, index):
        node_label = torch.zeros(self.graphs[index]['num_nodes'])
        node_label[self.activate_nodes[index]] = 1
        if self.rxn_class is not None:
            return self.graphs[index], self.rxn_class[index], node_label, \
                self.edge_edits[index], self.edge_types[index]
        else:
            return self.graphs[index], node_label, \
                self.edge_edits[index], self.edge_types[index]


class ExtendedBondEncoder(torch.nn.Module):
    def __init__(self, dim: int, n_class: Optional[int] = None):
        """The extened bond encoder, extended from ogb bond encoder for mol
        it padded the edge feat matrix with learnable padding embedding

        Args:
            dim (int): dim of output result and embedding table
            n_class (int): number of reaction classes (default: `None`)
        """
        super(ExtendedBondEncoder, self).__init__()
        self.padding_emb = torch.nn.Parameter(torch.zeros(dim))
        self.bond_encoder = BondEncoder(dim)
        self.dim, self.n_class = dim, n_class
        torch.nn.init.xavier_uniform_(self.padding_emb)

        if n_class is not None:
            self.class_emb = torch.nn.Embedding(n_class, dim)
            self.lin = torch.nn.Linear(dim + dim, dim)

    def get_padded_edge_feat(
        self, edge_index: torch.Tensor,
        edge_feat: torch.Tensor, num_nodes: int
    ) -> torch.Tensor:
        """[summary]

        Forward a batch for edge feats, using ogb bond encoder
        padding embedding and class embedding

        Args:
            edge_index (torch.Tensor): a tensor of shape [2, num_edges]
            edge_feat (torch.Tensor): tensor of shape [num_edges, feat_dim]
            num_nodes (int): number of nodes in graph

        Returns:
            torch.Tensor: a tensor of shape [num_nodes, num_nodes, feat_dim]
            the encoded result of bond encoder, padded with learnable 
            padding embedding
        """
        edge_result = self.padding_emb.repeat(num_nodes, num_nodes, 1)
        xindex, yindex = edge_index
        edge_result[xindex, yindex] = self.bond_encoder(edge_feat)
        return edge_result

    def forward(
        self, num_nodes: List[int], edge_index: List[torch.Tensor],
        edge_feat: List[torch.Tensor], rxn_class: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        given a batch of graph information: num of nodes of each graph,
        edge index of each graph and their egde features, return the padded
        edge embedding matrix

        Args:
            num_nodes (List[int]): a list containing the number of nodes of each graph
            edge_index (List[torch.Tensor]): list of edge indexes of each graph
            edge_feat (List[torch.Tensor]): edge feature for ogb bond encoder of each graph
            rxn_class (torch.Tensor): reaction class of each graph (default: `None`)

        Returns:
            torch.Tensor: a tensor of [batch_size, max_node, max_node, dim]
                the padded edge_feature for the following training 
        """
        max_node, batch_size = max(num_nodes), len(num_nodes)
        if self.n_class is not None:
            if rxn_class is None:
                raise ValueError('class information should be given')
            else:
                rxn_class_emb = self.class_emb(rxn_class)
                rxn_class_emb = rxn_class_emb.reshape(batch_size, 1, 1, -1)
                rxn_class_emb = rxn_class_emb.repeat(1, max_node, max_node, 1)

        edge_result = torch.zeros(batch_size, max_node, max_node, self.dim)
        edge_result = edge_result.to(edge_feat[0].device)

        for idx in range(batch_size):
            edge_matrix = self.get_padded_edge_feat(
                edge_index[idx], edge_feat[idx], num_nodes[idx]
            )
            edge_result[idx][:num_nodes[idx], num_nodes[idx]] = edge_matrix

        if self.n_class is not None:
            edge_result = torch.cat([edge_result, rxn_class_emb], dim=-1)
            edge_result = self.lin(edge_result)
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
    max_node = max([x[0]['num_nodes'] for x in data_batch])
    attn_mask = torch.zeros(batch_size, max_node, max_node, dtype=bool)
    node_label = torch.ones(batch_size, max_node) * -100
    edge_edits, edge_types, graphs, rxn_class = [], [], [], []
    for idx, data in enumerate(data_batch):
        if len(data) == 4:
            graph, n_lb, e_ed, e_type = data
        else:
            graph, r_class, n_lb, e_ed, e_type = data
            rxn_class.append(r_class)
        node_num = graph['num_nodes']
        node_label[idx][:node_num] = n_lb
        attn_mask[idx][:node_num, :node_num] = True
        edge_edits.append(e_ed)
        edge_types.append(e_type)
        graphs.append(graph)
    if len(rxn_class) == 0:
        return graphs, node_label, edge_edits, edge_types
    else:
        return graphs, torch.LongTensor(rxn_class),\
            node_label, edge_edits, edge_types
