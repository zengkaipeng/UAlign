import torch
import torch_scatter
from typing import Any, Dict, List, Tuple, Optional, Union
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from torch_geometric.data import Data


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
        dropout: float = 0.7,
        residual: bool = True
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
            self.edge_update.append(SparseEdgeUpdateLayer(
                embedding_dim, embedding_dim, residual=residual
            ))
        self.residual = residual

    def forward(
        self,
        node_feats: torch.Tensor, edge_feats: torch.Tensor,
        edge_index: torch.Tensor, num_nodes: int
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


class MixGraphEncoder(torch.nn.Modlue):
    def __init__(
        self,
        embedding_dim: int = 64,
        input_dim: Optional[int] = None
    ):
        super(MixGraphEncoder, self).__init__()
        self.atom_encoder = torch.nn.ModuleDict({
            'ogb': AtomEncoder(embedding_dim)
        })
        self.bond_encoder = torch.nn.ModuleDict({
            'ogb': BondEncoder(embedding_dim)
        })
        if input_dim is not None:
            self.atom_encoder['common'] = torch.nn.Linear(
                input_dim, embedding_dim
            )
            self.edge_encoder['common'] = torch.nn.Linear(
                embedding_dim * 2, embedding_dim
            )

    def forward(
        self,
        graph: Dict[str, Union[int, torch.Tensor]]
        mode: str = 'ogb'
    ) -> Dict[str, Union[int, torch.Tensor]]:
        x_ogb = self.atom_encoder['ogb'](graph['x'])
        edge_ogb = self.bond_encoder['ogb'](graph['edge_attr'])
        if mode == ogb:
            assert x_ogb.shape[0] == graph['num_nodes'],\
                'number of nodes mismatch the number of feats' +\
                ' please make sure you are using the right mode'

            return {
                'node_feats': x_ogb, 'edge_feats': edge_ogb,
                'edge_index': graph['edge_index'],
                'num_nodes': graph['num_nodes'],
                'batch': graph['batch']
            }

        assert 'common' in self.atom_encoder, \
            'Missing Model Definition for common graph encoding'

        x_common = self.atom_encoder['common'](common_graph['x'])

        x = torch.cat([x_ogb, x_common], dim=0)
        assert x.shape[0] == graph['num_nodes'], \
            'number of nodes mismatch the number of feats' + \
            ' please make sure num nodes included all nodes'

        x_i = torch.index_select(x, dim=0, index=graph['edge_common'][0])
        x_j = torch.index_select(x, dim=0, index=graph['edge_common'][1])
        x_sum = x_i + x_j
        x_diff = torch.abs(x_i - x_j)
        edge_common = self.bond_encoder['common'](
            torch.cat([x_sum, x_diff], dim=-1)
        )

        edge = torch.cat([edge_ogb, edge_common], dim=0)
        edge_index = torch.cat([
            graph['edge_index'], graph['edge_common']
        ], dim=-1)

        return {
            'node_feats': x, 'edge_feats': edge, batch: graph['batch'],
            'num_nodes': graph['num_nodes'], 'edge_index': edge_index,
        }


def sparse_edit_collect_fn(data_batch):
    batch_size, rxn_class, node_label, num_l = len(data_batch), [], [], []
    edge_idxes, edge_feats, node_feats, lstnode, batch = [], [], [], 0, []
    for idx, data in enumerate(data_batch):
        if len(data) == 4:
            graph, n_lb, e_ed, e_type = data
        else:
            graph, r_class, n_lb, e_ed, e_type = data
            rxn_class.append(r_class)

        edge_edits.append(e_ed)
        edge_types.append(e_type)
        node_label.append(n_lb)
        num_l.append(graph['num_nodesobject: _T'])

        edge_idxes.append(graph['edge_index'] + lstnode)
        edge_feats.append(graph['edge_feat'])
        node_feats.append(graph['node_feat'])
        lstnode += graph['num_nodes']
        batch.append(np.ones(graph['num_nodes'], dtype=np.int64) * idx)

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
        return Data(**result), node_label, num_l, edge_edits, edge_types
    else:
        return Data(**result), torch.LongTensor(rxn_class),\
            node_label, num_l, edge_edits, edge_types
