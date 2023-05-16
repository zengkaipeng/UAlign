import torch
from typing import Any, Dict, List, Tuple, Optional, Union
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from torch_geometric.data import Data


class EditDataset(torch.utils.data.Dataset):
    def __init__(
        self, graphs: List[Dict],
        activate_nodes: List[List],
        edge_types: List[List[List]],
        rxn_class: Optional[List[int]] = None
    ):
        super(EditDataset, self).__init__()
        self.graphs = graphs
        self.activate_nodes = activate_nodes
        self.edge_types = edge_types
        self.rxn_class = rxn_class

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, index):
        node_label = torch.zeros(self.graphs[index]['num_nodes']).long()
        node_label[self.activate_nodes[index]] = 1
        if self.rxn_class is not None:
            return self.graphs[index], self.rxn_class[index], node_label, \
                self.edge_types[index], self.activate_nodes[index]
        else:
            return self.graphs[index], node_label, \
                self.edge_types[index], self.activate_nodes[index]


class ExtendedBondEncoder(torch.nn.Module):
    def __init__(self, dim: int):
        """The extened bond encoder, extended from ogb bond encoder for mol
        it padded the edge feat matrix with learnable padding embedding

        Args:
            dim (int): dim of output result and embedding table
        """
        super(ExtendedBondEncoder, self).__init__()
        self.padding_emb = torch.nn.Parameter(torch.randn(dim))
        self.self_loop = torch.nn.Parameter(torch.randn(dim))
        self.bond_encoder = BondEncoder(dim)
        self.dim = dim

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
        diag_index = list(range(num_nodes))
        edge_result[diag_index, diag_index] = self.self_loop
        return edge_result

    def forward(
        self, num_nodes: List[int], edge_index: List[torch.Tensor],
        edge_feat: List[torch.Tensor]
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

        edge_result = torch.zeros(batch_size, max_node, max_node, self.dim)
        edge_result = edge_result.to(edge_feat[0].device)

        for idx in range(batch_size):
            edge_matrix = self.get_padded_edge_feat(
                edge_index[idx], edge_feat[idx], num_nodes[idx]
            )
            edge_result[idx][:num_nodes[idx], :num_nodes[idx]] = edge_matrix

        return edge_result


class ExtendedAtomEncoder(torch.nn.Module):
    def __init__(self, dim: int):
        """The extened bond encoder, extended from ogb atom encoder for mol
        it padded the atom feature matrix with zero features.

        Args:
            dim (int): dim of output result and embedding table
        """
        super(ExtendedAtomEncoder, self).__init__()
        self.atom_encoder = AtomEncoder(dim)
        self.dim = dim

    def forward(
        self, num_nodes: List[int], node_feat: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        given a batch of graph informations, num_nodes and node_feats
        and class type (optional) return a matrix of encoded graph feat

        Args:
            num_nodes (List[int]): a list of batch size, containing 
                the number of nodes of each graph
            node_feat (List[torch.Tensor]): a list of tensors, each 
                is of the shape [num_nodes, feat_dim]

        Returns:
            torch.Tensor: a tensor of shape [batch_size, num_node, dim]
            representing the padded node features, padded with zeros.
        """
        max_node, batch_size = max(num_nodes), len(num_nodes)
        result = torch.zeros(batch_size, max_node, self.dim)
        result = result.to(node_feat[0].device)
        for idx in range(batch_size):
            result[idx][:num_nodes[idx]] = self.atom_encoder(node_feat[idx])

        return result


class EdgeUpdateLayer(torch.nn.Module):
    def __init__(
        self,  edge_dim: int = 64, node_dim: int = 64,
        residual: bool = False, use_ln: bool = True
    ):
        super(EdgeUpdateLayer, self).__init__()
        input_dim = node_dim * 2 + edge_dim
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(input_dim, input_dim),
            torch.nn.LayerNorm(input_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(input_dim, edge_dim)
        )
        self.residual = residual
        self.use_ln = use_ln
        if self.use_ln:
            self.ln_norm1 = torch.nn.LayerNorm(edge_dim)

    def forward(
        self, node_feat: torch.Tensor, edge_feats: torch.Tensor,
    ) -> torch.Tensor:
        node_feat1 = torch.unsqueeze(node_feat, dim=1)
        node_feat2 = torch.unsqueeze(node_feat, dim=2)
        batch_size, num_nodes, dim = node_feat.shape
        node_feat1 = node_feat1.repeat(1, num_nodes, 1, 1)
        node_feat2 = node_feat2.repeat(1, 1, num_nodes, 1)
        x = torch.cat([node_feat1, node_feat2, edge_feats], dim=-1)
        result = self.mlp(x) + edge_feats if self.residual else self.mlp(x)
        if self.use_ln:
            return self.ln_norm1(result)
        else:
            return result


class FCGATLayer(torch.nn.Module):
    def __init__(
        self, input_dim: int, edge_dim: int, n_heads: int,
        dropout: float = 0.1, negative_slope: float = 0.1
    ):
        super(FCGATLayer, self).__init__()
        output_dim = input_dim // n_heads
        assert output_dim * n_heads == input_dim, \
            "input dim should be evenly divided by n_heads"
        self.input_dim, self.output_dim = input_dim, output_dim
        self.negative_slope, self.n_heads = negative_slope, n_heads
        self.dropout_fun = torch.nn.Dropout(dropout)
        self.edge_lin = torch.nn.Linear(edge_dim, output_dim * n_heads, False)
        self.lin_V = torch.nn.Linear(input_dim, output_dim * n_heads, False)
        self.lin_message = torch.nn.Linear(edge_dim, output_dim * n_heads)
        attn_node_shape = (1, 1, n_heads, output_dim)
        attn_edge_shape = (1, 1, 1, n_heads, output_dim)
        self.att_src = torch.nn.Parameter(torch.zeros(*attn_node_shape))
        self.att_dst = torch.nn.Parameter(torch.zeros(*attn_node_shape))
        self.att_edge = torch.nn.Parameter(torch.zeros(*attn_edge_shape))
        self.bias_node = torch.nn.Parameter(torch.zeros(n_heads, output_dim))

        torch.nn.init.xavier_uniform_(self.att_src)
        torch.nn.init.xavier_uniform_(self.att_dst)
        torch.nn.init.xavier_uniform_(self.att_edge)
        self.ff_block = torch.nn.Sequential(
            torch.nn.Linear(n_heads * output_dim, n_heads * output_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(n_heads * output_dim, n_heads * output_dim),
            torch.nn.Dropout(dropout)
        )

        self.ln_norm1 = torch.nn.LayerNorm(output_dim * n_heads)
        self.ln_norm2 = torch.nn.LayerNorm(output_dim * n_heads)

    def forward(
        self, node_feats: torch.Tensor, edge_feats: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        forward function, input the node feats and edge feats, 
        output the node feat of a new layer
        Args:
            node_feats (torch.Tensor): a tensor of shape [batch_size, max_node, dim]
            edge_feats (torch.Tensor): a tensor of shape [batch_size, max_node, max_node, dim]
            attn_mask (torch.Tensor): a tensor of shape [batch_size, max_node, max_node]
            the place set as false will not be calculated as attn_weight (default: `None`)
        """
        batch_size, max_node = node_feats.shape[:2]
        # print('[INFO] Fshape', node_feats.shape, edge_feats.shape)
        node_o_shape = [batch_size, -1, self.n_heads, self.output_dim]
        edge_o_shape = [
            batch_size, max_node, max_node,
            self.n_heads, self.output_dim
        ]
        x_proj = self.lin_V(node_feats).reshape(*node_o_shape)
        src_att = (self.att_src * x_proj).sum(dim=-1)
        dst_att = (self.att_dst * x_proj).sum(dim=-1)
        src_att = src_att.unsqueeze(dim=2).repeat(1, 1, max_node, 1)
        dst_att = dst_att.unsqueeze(dim=1).repeat(1, max_node, 1, 1)

        e_value = self.edge_lin(edge_feats).reshape(*edge_o_shape)
        e_message = self.lin_message(edge_feats).reshape(*edge_o_shape)
        e_att = (self.att_edge * e_value).sum(dim=-1)

        att_weight = torch.nn.functional.leaky_relu(
            src_att + dst_att + e_att, self.negative_slope
        )

        # print('[INFO] Ashape', attn_mask.shape, att_weight.shape)
        if attn_mask is not None:
            attn_mask = torch.logical_not(attn_mask.unsqueeze(dim=-1))
            INF = (1 << 32) - 1
            att_weight = torch.masked_fill(att_weight, attn_mask, -INF)
        att_weight = torch.softmax(att_weight, dim=2).unsqueeze(-1)
        # [batch_size, max_node, max_node, n_heads, 1]

        x_v = x_proj.unsqueeze(dim=1).repeat(1, max_node, 1, 1, 1)

        x_output = att_weight * (e_message + x_v)
        x_output = self.dropout_fun(x_output.sum(dim=2) + self.bias_node)
        x_output = x_output.reshape(batch_size, max_node, -1)

        x_output = self.ln_norm1(node_feats + x_output)
        x_output = self.ln_norm2(x_output + self.ff_block(x_output))
        return x_output


class FCGATEncoder(torch.nn.Module):
    def __init__(
        self, n_layers: int = 6, n_heads: int = 4,
        embedding_dim: int = 256, dropout: float = 0.2,
        negative_slope: float = 0.2, edge_last: bool = False
    ):
        super(FCGATEncoder, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.edge_update = torch.nn.ModuleList()
        self.num_layers, self.num_heads = n_layers, n_heads
        for layer in range(self.num_layers):
            self.convs.append(FCGATLayer(
                embedding_dim, embedding_dim, n_heads,
                dropout=dropout, negative_slope=negative_slope
            ))
            if edge_last or layer < self.num_layers - 1:
                self.edge_update.append(EdgeUpdateLayer(
                    embedding_dim, embedding_dim, residual=True,
                    use_ln=True
                ))
        self.drop_fun = torch.nn.Dropout(dropout)
        self.edge_last = edge_last

    def forward(
        self,
        node_feats: torch.Tensor, edge_feats: torch.Tensor,
        attn_mask: torch.Tensor
    ):
        for layer in range(self.num_layers):
            node_feats = self.convs[layer](
                node_feats=node_feats, edge_feats=edge_feats,
                attn_mask=attn_mask
            )
            if self.edge_last or layer < self.num_layers - 1:
                edge_feats = self.edge_update[layer](
                    node_feat=node_feats, edge_feats=edge_feats
                )

        return node_feats, edge_feats


def edit_collect_fn(data_batch):
    batch_size = len(data_batch)
    max_node = max([x[0]['num_nodes'] for x in data_batch])
    attn_mask = torch.zeros(batch_size, max_node, max_node, dtype=bool)
    node_label = torch.ones(batch_size, max_node) * -100
    edge_cores, edge_types, graphs, rxn_class = [], [], [], []
    for idx, data in enumerate(data_batch):
        if len(data) == 4:
            graph, n_lb, e_type, e_core = data
        else:
            graph, r_class, n_lb, e_type, e_core = data
            rxn_class.append(r_class)
        node_num = graph['num_nodes']
        node_label[idx][:node_num] = n_lb
        attn_mask[idx][:node_num, :node_num] = True
        edge_cores.append(e_core)
        edge_types.append(e_type)

        graph['node_feat'] = torch.from_numpy(graph['node_feat']).float()
        graph['edge_feat'] = torch.from_numpy(graph['edge_feat']).float()
        graph['edge_index'] = torch.from_numpy(graph['edge_index'])
        graphs.append(Data(**graph))

    if len(rxn_class) == 0:
        return attn_mask, graphs, node_label, edge_cores, edge_types
    else:
        return attn_mask, graphs, torch.LongTensor(rxn_class),\
            node_label, edge_cores, edge_types
