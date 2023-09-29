import torch


class MyGINConv(torch.nn.Module):
    def __init__(self, embedding_dim: int = 64):
        super(MyGINConv, self).__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim, 2 * embedding_dim),
            torch.nn.LayerNorm(2 * embedding_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(2 * embedding_dim, embedding_dim)
        )
        self.eps = torch.nn.Parameter(torch.Tensor([0]))

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        num_nodes, (row, col) = x.shape[0], edge_index
        message_node = torch.index_select(input=x, dim=0, index=col)
        message = torch.relu(message_node + edge_attr)
        dim = message.shape[-1]

        message_reduce = torch.zeros(num_nodes, dim).to(message)
        message_reduce.index_add_(dim=0, index=row, source=message)

        return self.mlp((1 + self.eps) * x + torch.relu(message_reduce))
