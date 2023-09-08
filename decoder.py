import torch
from sparse_backBone import SparseAtomEncoder, SparseBondEncoder
from Mix_backbone import MhAttnBlock


class Feat_init(torch.nn.Module):
    def __init__(
        self, n_pad, dim, heads=2, dropout=0.1, negative_slope=0.2
    ):
        super(Feat_init, self).__init__()
        self.Qemb = torch.nn.Parameter(torch.randn(1, n_pad, dim))
        self.atom_encoder = SparseAtomEncoder(dim)
        self.bond_encoder = SparseBondEncoder(dim)
        self.Attn = MhAttnBlock(
            Qdim=dim, Kdim=dim, Vdim=dim, Odim=dim,
            heads=heads, dropout=dropout
        )
        self.dim = dim
        self.edge_lin = torch.nn.Linear(dim * 2, dim)

    def forward(self, graph, memory, cross_mask):
        device, batch_size = graph.x.device, graph.batch.max().item() + 1

        # get node feat
        node_feat = torch.zeros((graph.num_nodes, self.dim)).to(device)
        org_node_feat = self.atom_encoder(graph.x, graph.get('node_rxn', None))
        node_feat[graph.node_org_mask] = org_node_feat

        pad_node_feat = self.Attn(
            Q=self.Qemb.repeat(batch_size, 1, 1), K=memory,
            V=memory, attn_mask=cross_mask
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
