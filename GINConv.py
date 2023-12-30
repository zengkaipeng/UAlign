import torch
from typing import Optional


class MyGINConv(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, edge_dim: int):
        super(MyGINConv, self).__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(in_channels, 2 * in_channels),
            torch.nn.LayerNorm(2 * in_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(2 * in_channels, out_channels)
        )
        self.eps = torch.nn.Parameter(torch.Tensor([0]))
        self.padding_edge = torch.nn.Parameter(torch.randn(edge_dim))
        self.lin_edge = torch.nn.Linear(edge_dim, in_channels)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_dim = edge_dim

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        num_nodes = x.shape[0]
        message = torch.relu(x[edge_index[1]] + self.lin_edge(edge_attr))
        message_reduce = torch.zeros(num_nodes, self.in_channels).to(message)
        message_reduce.index_add_(dim=0, index=edge_index[0], source=message)

        return self.mlp((1 + self.eps) * x + message_reduce)