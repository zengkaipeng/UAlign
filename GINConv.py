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
        self,
        node_feats: torch.Tensor, edge_feats: torch.Tensor,
        edge_index: torch.Tensor, num_nodes: int
    ) -> torch.Tensor:
        message_node = torch.index_select(
            input=node_feats, dim=0, index=edge_index[1]
        )
        message = torch.relu(message_node + edge_feats)

        dim = message.shape[-1]

        message_reduce = torch.zeros(num_nodes, dim).to(message)
        index = edge_index[0].unsqueeze(-1).repeat(1, dim)
        message_reduce.scatter_add_(dim=0, index=index, src=message)

        return self.mlp((1 + self.eps) * node_feats + message_reduce)
