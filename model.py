import torch
import torch_scatter
from ogb.utils import smiles2graph
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder


class GINConv(torch.nn.Module):
    def __init__(self, embedding_dim: int = 64):
        super(GINConv, self).__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim, 2 * embedding_dim),
            torch.nn.BatchNorm1d(2 * embedding_dim),
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


class GINBase(torch.nn.Module):
    def __init__(
        self,  num_layers: int = 4,
        embedding_dim: int = 64,
        dropout: float = 0.7
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
            self.batch_norms.append(torch.nn.BatchNorm1d(embedding_dim))
            self.edge_update.append(
            	EdgeUpdateLayer(embedding_dim, embedding_dim)
            )

    def forward(
        self,
        node_feats: torch.Tensor, edge_feats: torch.Tensor,
        edge_index: torch.Tensor, num_nodes: int, num_edges: int
    ) -> torch.Tensor:
    	for layer in range(self.num_layers):
    		node_feats = self.bathc_norms[layer](self.convs[layer](
    			node_feats=node_feats, edge_feats=edge_feats,
    			edge_index=edge_index, num_nodes=num_nodes,
    		))

    		edge_feats = self.edge_update[layer](
    			edge_feats=edge_feats, node_feats=node_feats,
    			edge_index=edge_index
    		)

            node_feats = self.dropout_fun(
            	node_feats if layer == self.num_layers - 1
            	else torch.relu(node_feats)
            )
        return node_feats, edge_feats



