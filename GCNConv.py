import torch


class MyGCNConv(torch.nn.Module):
    def __init__(self, emb_dim):
        super(MyGCNConv, self).__init__()
        self.root_emb = torch.nn.Parameter(torch.randn(emb_dim))
        self.linear = torch.nn.Linear(emb_dim, emb_dim)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        x = self.linear(x)
        (N, Dim), device = x.shape, x.device
        deg_i = torch.ones(N).to(device)
        deg_j = torch.ones(N).to(device)

        (row, col), num_edge = edge_index, edge_index.shape[1]

        deg_i.index_add_(
            dim=0, index=row,
            source=torch.ones(num_edge).to(device)
        )
        deg_j.index_add_(
            dim=0, index=col,
            source=torch.ones(num_edge).to(device)
        )

        deg_inv_sqrt_i = deg_i ** (-0.5)
        deg_inv_sqrt_j = deg_j ** (-0.5)

        adj_t_val = torch.index_select(deg_inv_sqrt_i, dim=0, index=row)\
            * torch.index_select(deg_inv_sqrt_j, dim=0, index=col)

        adj_t = torch.sparse_coo_tensor(edge_index, adj_t_val, size=(N, N))

        message_edge = torch.zeros_like(x).to(x.device)
        message_edge.scatter_add_(
            dim=0, src=adj_t_val.unsqueeze(-1) * edge_attr,
            index=row.unsqueeze(-1).repeat(1, Dim)
        )

        node_feat = torch.relu(torch.matmul(adj_t, x) + message_edge) + \
            torch.relu(x + self.root_emb) * \
            (deg_inv_sqrt_i * deg_inv_sqrt_j).unsqueeze(-1)

        return node_feat
