import torch
from sparse_backBone import (
    SparseAtomEncoder, SparseBondEncoder, SparseEdgeUpdateLayer
)
from Mix_backbone import MhAttnBlock
from GINConv import MyGINConv
from GATconv import NyGATConv
from GCNConv import MyGCNConv


def batch_mask(
    ptr: torch.Tensor, max_node: int, batch_size: int
) -> torch.Tensor:
    num_nodes = ptr[1:] - ptr[:-1]
    mask = torch.arange(max_node).repeat(batch_size, 1)
    mask = mask.to(num_nodes.device)
    return mask < num_nodes.reshape(-1, 1)


def graph2batch(
    node_feat: torch.Tensor, batch_mask: torch.Tensor,
    batch_size: int, max_node: int
) -> torch.Tensor:
    answer = torch.zeros(batch_size, max_node, node_feat.shape[-1])
    answer = answer.to(node_feat.device)
    answer[batch_mask] = node_feat
    return answer


class Feat_init(torch.nn.Module):
    def __init__(
        self, n_pad: int, dim: int, heads: int = 2, dropout: float = 0.1,
        negative_slope: float = 0.2, n_class: Optional[int] = None
    ):
        super(Feat_init, self).__init__()
        self.Qemb = torch.nn.Parameter(torch.randn(1, n_pad, dim))
        self.atom_encoder = SparseAtomEncoder(dim, n_class)
        self.bond_encoder = SparseBondEncoder(dim, n_class)

        assert dim % heads == 0, 'dim should be evenly divided by heads'

        if n_class is not None:
            self.cls_emb = torch.nn.Embedding(n_class, dim)
            self.Attn = MhAttnBlock(
                Qdim=2 * dim, Kdim=dim, Vdim=dim,
                Odim=dim // heads, heads=heads, dropout=dropout
            )
        else:
            self.Attn = MhAttnBlock(
                Qdim=dim, Kdim=dim, Vdim=dim, Odim=dim // heads,
                heads=heads, dropout=dropout
            )

        self.dim, self.n_class, self.n_pad = dim, n_class, n_pad
        self.edge_lin = torch.nn.Linear(dim * 2, dim)

    def forward(self, graph, memory, mem_pad_mask=None):
        device, batch_size = graph.x.device, graph.batch.max().item() + 1

        # get node feat
        node_feat = torch.zeros((graph.num_nodes, self.dim)).to(device)
        org_node_feat = self.atom_encoder(graph.x, graph.get('node_rxn', None))
        node_feat[graph.node_org_mask] = org_node_feat

        if self.n_class is not None:
            emb_cls = self.cls_emb(graph.graph_rxn)  # [batch_size, dim]
            emb_cls = emb_cls.unsqueeze(dim=0).repeat(1, self.n_pad, 1)
            Qval = self.Qemb.repeat(batch_size, 1, 1)
            Qval = torch.cat([Qval, emb_cls], dim=-1)
        else:
            Qval = self.Qemb.repeat(batch_size, 1, 1)

        if mem_pad_mask is not None:
            attn_mask = mem_pad_mask.unsqueeze(1).repeat(1, self.n_pad, 1)
        else:
            attn_mask = None

        pad_node_feat = self.Attn(
            Q=Qval, K=memory, V=memory, attn_mask=attn_mask
        )
        # [B, pad, dim]
        node_feat[graph.node_pad_mask] = pad_node_feat.reshape(-1, self.dim)

        edge_feat = self.bond_encoder(
            graph.edge_attr, graph.org_mask, graph.self_mask,
            graph.get('edge_rxn', None)
        )

        pad_i, pad_j = graph.edge_index[:, graph.pad_mask]

        pad_edge_feat = torch.cat([node_feat[pad_i], node_feat[pad_j]], dim=-1)
        pad_edge_feat = self.edge_lin(torch.relu(pad_edge_feat))
        edge_feat[graph.pad_mask] = pad_edge_feat

        return node_feat, edge_feat


class MixDecoder(torch.nn.Module):
    def __init__(
        self, emb_dim: int, n_layers: int, gnn_args: Union[Dict, List[Dict]],
        n_pad: int, dropout: float = 0, heads: int = 1, gnn_type: str = 'gin',
        negative_slope: float = 0.2, n_class: Optional[int] = None,
        update_gate: str = 'add'
    ):
        super(MixDecoder, self).__init__()

        self.feat_init = Feat_init(
            n_pad, emb_dim, heads=heads, dropout=dropout,
            n_class=n_class, negative_slope=negative_slope
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
            self.cross_attns.append(MhAttnBlock(
                Qdim=dim, Kdim=dim, Vdim=dim, Odim=dim // heads,
                heads=heads, dropout=dropout
            ))

    def forward(self, graph, memory, mem_pad_mask=None):
        node_feats, edge_feats = self.feat_init(graph, memory, mem_pad_mask)

        batch_size, max_node = graph.attn_mask.shape[:2]
        batch_mask = batch_mask(graph.ptr, max_node, batch_size)
        if mem_pad_mask is not None:
            cross_mask = torch.zeros_like(mem_pad_mask)
            cross_mask[mem_pad_mask] = True
            cross_mask = cross_mask.unsqueeze(1).repeat(1, max_node, 1)
            cross_mask[~batch_mask] = False
        else:
            cross_mask = None

        for i in range(self.num_layers):
            conv_res = self.convs[i](
                node_feat=node_feats, edge_feat=edge_feats, ptr=graph.ptr,
                attn_mask=graph.attn_mask, edge_index=graph.edge_index,
            ) + node_feats

            node_feats = self.dropout_fun(torch.relu(self.lns[i](conv_res)))

            node_feats = graph2batch(
                node_feats, batch_mask, batch_size, max_node
            )

            cross_res = self.cross_attns[i](
                Q=node_feats, K=memory, V=memory, attn_mask=cross_mask
            ) + node_feats

            node_feats = torch.relu(self.ln2[i](cross_res))[batch_mask]

            edge_feats = self.edge_update[i](
                edge_feats=edge_feats, node_feats=node_feats,
                edge_index=graph.edge_index
            )

        return node_feats, edge_feats


class GATDecoder(torch.nn.Module):
    def __init__(
        self, num_layers: int = 4, num_heads: int = 4, embedding_dim: int = 64,
        dropout: float = 0.7,  self_loop: bool = True,
        negative_slope: float = 0.2, n_class: Optional[int] = None
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
                negative_slope=negative_slope, add_self_loop=self_loop,
                dropout=dropout, edge_dim=embedding_dim
            ))
            self.ln1.append(torch.nn.LayerNorm(embedding_dim))
            self.ln2.append(torch.nn.LayerNorm(embedding_dim))
            self.edge_update.append(SparseEdgeUpdateLayer(
                embedding_dim, embedding_dim, residual=True
            ))
            self.cross_attns.append(MhAttnBlock(
                Qdim=dim, Kdim=dim, Vdim=dim, Odim=dim // heads,
                heads=heads, dropout=dropout
            ))
        self.add_self_loop = self_loop

        self.feat_init = Feat_init(
            n_pad, emb_dim, heads=heads, dropout=dropout,
            n_class=n_class, negative_slope=negative_slope
        )

    def forward(self, graph) -> torch.Tensor:
        node_feats, edge_feats = self.feat_init(graph, memory, mem_pad_mask)
        batch_size, max_node = graph.attn_mask.shape[:2]
        batch_mask = batch_mask(graph.ptr, max_node, batch_size)
        if mem_pad_mask is not None:
            cross_mask = torch.zeros_like(mem_pad_mask)
            cross_mask[mem_pad_mask] = True
            cross_mask = cross_mask.unsqueeze(1).repeat(1, max_node, 1)
            cross_mask[~batch_mask] = False
        else:
            cross_mask = None

        for layer in range(self.num_layers):
            conv_res = self.ln1[layer](self.convs[layer](
                x=node_feats, edge_attr=edge_feats,
                edge_index=graph.edge_index
            ))
            node_feats = self.dropout_fun(torch.relu(conv_res)) + node_feats

            node_feats = graph2batch(
                node_feats, batch_mask, batch_size, max_node
            )

            cross_res = self.cross_attns[i](
                Q=node_feats, K=memory, V=memory, attn_mask=cross_mask
            ) + node_feats

            node_feats = torch.relu(self.ln2[i](cross_res))[batch_mask]

            edge_feats = self.edge_update[layer](
                edge_feats=edge_feats, node_feats=node_feats,
                edge_index=graph.edge_index
            )

        return node_feats, edge_feats


class GCNDecoder(torch.nn.Module):
    def __init__(
        self,  num_layers: int = 4,
        embedding_dim: int = 64,
        dropout: float = 0.7,
        n_class: Optional[int] = None
    ):
        super(GCNDecoder, self).__init__()
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
            self.convs.append(MyGCNConv(embedding_dim))
            self.ln1.append(torch.nn.LayerNorm(embedding_dim))
            self.ln2.append(torch.nn.LayerNorm(embedding_dim))
            self.edge_update.append(SparseEdgeUpdateLayer(
                embedding_dim, embedding_dim, residual=True
            ))
            self.cross_attns.append(MhAttnBlock(
                Qdim=dim, Kdim=dim, Vdim=dim, Odim=dim // heads,
                heads=heads, dropout=dropout
            ))
        self.feat_init = Feat_init(
            n_pad, emb_dim, heads=heads, dropout=dropout,
            n_class=n_class, negative_slope=negative_slope
        )

    def forward(self, graph) -> torch.Tensor:
        node_feats, edge_feats = self.feat_init(graph, memory, mem_pad_mask)
        batch_size, max_node = graph.attn_mask.shape[:2]
        batch_mask = batch_mask(graph.ptr, max_node, batch_size)
        if mem_pad_mask is not None:
            cross_mask = torch.zeros_like(mem_pad_mask)
            cross_mask[mem_pad_mask] = True
            cross_mask = cross_mask.unsqueeze(1).repeat(1, max_node, 1)
            cross_mask[~batch_mask] = False
        else:
            cross_mask = None

        for layer in range(self.num_layers):
            conv_res = self.ln1[layer](self.convs[layer](
                x=node_feats, edge_attr=edge_feats,
                edge_index=graph.edge_index
            ))
            node_feats = self.dropout_fun(conv_res) + node_feats
            node_feats = graph2batch(
                node_feats, batch_mask, batch_size, max_node
            )
            cross_res = self.cross_attns[i](
                Q=node_feats, K=memory, V=memory, attn_mask=cross_mask
            ) + node_feats
            node_feats = torch.relu(self.ln2[i](cross_res))[batch_mask]

            edge_feats = self.edge_update[layer](
                edge_feats=edge_feats, node_feats=node_feats,
                edge_index=graph.edge_index
            )
        return node_feats, edge_feats


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
        self.ln1 = torch.nn.ModuleList()
        self.ln2 = torch.nn.ModuleList()
        self.cross_attns = torch.nn.ModuleList()
        self.edge_update = torch.nn.ModuleList()
        self.num_layers = num_layers
        self.dropout_fun = torch.nn.Dropout(dropout)
        for layer in range(self.num_layers):
            self.convs.append(MyGINConv(embedding_dim))
            self.ln1.append(torch.nn.LayerNorm(embedding_dim))
            self.ln2.append(torch.nn.LayerNorm(embedding_dim))
            self.edge_update.append(SparseEdgeUpdateLayer(
                embedding_dim, embedding_dim, residual=True
            ))
        self.feat_init = Feat_init(
            n_pad, emb_dim, heads=heads, dropout=dropout,
            n_class=n_class, negative_slope=negative_slope
        )

    def forward(self, graph) -> torch.Tensor:
        node_feats, edge_feats = self.feat_init(graph, memory, mem_pad_mask)
        batch_size, max_node = graph.attn_mask.shape[:2]
        batch_mask = batch_mask(graph.ptr, max_node, batch_size)
        if mem_pad_mask is not None:
            cross_mask = torch.zeros_like(mem_pad_mask)
            cross_mask[mem_pad_mask] = True
            cross_mask = cross_mask.unsqueeze(1).repeat(1, max_node, 1)
            cross_mask[~batch_mask] = False
        else:
            cross_mask = None

        for layer in range(self.num_layers):
            conv_res = self.ln1[layer](self.convs[layer](
                x=node_feats, edge_attr=edge_feats,
                edge_index=graph.edge_index
            ))
            node_feats = self.dropout_fun(torch.relu(conv_res)) + node_feats
            node_feats = graph2batch(
                node_feats, batch_mask, batch_size, max_node
            )
            cross_res = self.cross_attns[i](
                Q=node_feats, K=memory, V=memory, attn_mask=cross_mask
            ) + node_feats
            node_feats = torch.relu(self.ln2[i](cross_res))[batch_mask]

            edge_feats = self.edge_update[layer](
                edge_feats=edge_feats, node_feats=node_feats,
                edge_index=graph.edge_index
            )
        return node_feats, edge_feats
