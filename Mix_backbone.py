import torch
from typing import Any, Dict, List, Tuple, Optional, Union
from collections.abc import Iterable
import torch.nn.functional as F
from GATconv import SelfLoopGATConv as MyGATConv
from GINConv import MyGINConv
from sparse_backBone import (
    SparseAtomEncoder, SparseBondEncoder, SparseEdgeUpdateLayer
)


class MixConv(torch.nn.Module):
    def __init__(
        self, emb_dim: int, gnn_args: Dict[str, Any], heads: int = 1,
        dropout: float = 0, gnn_type: str = 'gin', update_gate: str = 'add'
    ):
        super(MixConv, self).__init__()
        self.attn_conv = torch.nn.MultiheadAttention(
            emb_dim, batch_first=True, dropout=dropout, num_heads=heads,
        )
        assert update_gate in ['add', 'cat'], \
            f'Invalid update method {update_gate}'
        if gnn_type == 'gin':
            self.gnn_conv = MyGINConv(**gnn_args)
        elif gnn_type == 'gat':
            self.gnn_conv = MyGATConv(**gnn_args)
        else:
            raise NotImplementedError(f'Invalid gnn type {gcn}')
        self.update_method = update_gate
        if self.update_method == 'cat':
            self.update_gate = torch.nn.Linear(emb_dim << 1, emb_dim)
        self.heads = heads

    def forward(
        self, node_feat: torch.Tensor, edge_index: torch.Tensor,
        edge_feat: torch.Tensor, batch_mask: torch.Tensor,
        org_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        conv_res = self.gnn_conv(
            x=node_feat, edge_attr=edge_feat,
            edge_index=edge_index, org_mask=org_mask
        )
        batch_size, max_node = batch_mask.shape
        attn_input = torch.zeros(batch_size, max_node, node_feat.shape[-1])
        attn_input = attn_input.to(node_feat)
        attn_input[batch_mask] = node_feat

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.heads, 1, 1)
            attn_mask = attn_mask.reshape(-1, max_node, max_node)

        attn_res, attn_w = self.attn_conv(
            query=attn_input, key=attn_input, value=attn_input,
            attn_mask=attn_mask, key_padding_mask=~batch_mask,
        )

        if self.update_method == 'cat':
            return self.update_gate(torch.cat(
                [attn_res[batch_mask], conv_res], dim=-1
            ))
        else:
            return attn_res[batch_mask] + conv_res


class MixFormer(torch.nn.Module):
    def __init__(
        self, emb_dim: int, n_layers: int, gnn_args: Union[Dict, List[Dict]],
        dropout: float = 0, heads: int = 1,  n_class: Optional[int] = None,
        gnn_type: str = 'gin', update_gate: str = 'add',
    ):
        super(MixFormer, self).__init__()

        self.atom_encoder = SparseAtomEncoder(emb_dim, n_class)
        self.bond_encoder = SparseBondEncoder(emb_dim, n_class)

        self.num_layers = n_layers
        self.lns = torch.nn.ModuleList()
        self.convs = torch.nn.ModuleList()
        self.edge_update = torch.nn.ModuleList()

        self.dropout_fun = torch.nn.Dropout(dropout)

        for i in range(self.num_layers):
            self.lns.append(torch.nn.LayerNorm(emb_dim))
            gnn_layer = gnn_args[i] if isinstance(gnn_args, list) else gnn_args
            self.convs.append(MixConv(
                emb_dim=emb_dim, gnn_args=gnn_layer, heads=heads,
                dropout=dropout, gnn_type=gnn_type, update_gate=update_gate
            ))
            self.edge_update.append(SparseEdgeUpdateLayer(
                emb_dim, emb_dim, residual=True
            ))

    def forward(self, G):
        node_feats = self.atom_encoder(G.x, G.get('node_rxn', None))
        edge_feats = self.bond_encoder(G.edge_attr, G.get('edge_rxn', None))
        for i in range(self.num_layers):
            conv_res = self.convs[i](
                node_feat=node_feats, edge_feat=edge_feats,
                edge_index=G.edge_index, batch_mask=G.batch_mask,
                attn_mask=G.get('attn_mask', None),
                org_mask=G.get('e_org_mask', None)
            ) + node_feats

            node_feats = self.dropout_fun(torch.relu(self.lns[i](conv_res)))

            if G.get('e_org_mask', None) is not None:
                useful_edges = G.edge_index[:, G.e_org_mask]
            else:
                useful_edges = G.edge_index

            edge_feats = self.edge_update[i](
                edge_feats=edge_feats, node_feats=node_feats,
                edge_index=useful_edges
            )

        return node_feats, edge_feats
