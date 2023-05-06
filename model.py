import torch
from backBone import FCGATEncoder, ExtendedAtomEncoder, ExtendedBondEncoder
from sparse_backBone import GINBase, GATBase
from itertools import combinations


class GraphEditModel(torch.nn.Module):
    def __init__(
        self, base_model, is_sparse, node_dim, edge_dim,
        edge_class,  dropout=0.1
    ):
        super(GraphEditModel, self).__init__()
        self.base_model = base_model
        self.sparse = is_sparse

        if self.sparse:
            self.edge_feat_agger = torch.nn.Linear(
                node_dim + node_dim, edge_dim
            )
        self.edge_predictor = torch.nn.Sequential(
            torch.nn.Linear(edge_dim, edge_dim),
            torch.nn.Dropout(dropout),
            torch.nn.ReLU(),
            torch.nn.Linear(edge_dim, edge_class)
        )

        self.node_predictor = torch.nn.Sequential(
            torch.nn.Linear(node_dim, node_dim),
            torch.nn.Dropout(dropout),
            torch.nn.ReLU(),
            torch.nn.Linear(node_dim, 1)
        )
        if self.sparse:
            pass
        else:
            self.atom_encoder = ExtendedAtomEncoder(node_dim)
            self.edge_encoder = ExtendedBondEncoder(edge_dim)

    def get_edge_feat(self, node_feat, edge_index):
        assert self.sparse, 'Only sparse mode have edge_feat_agger'
        src_x = torch.index_select(node_feat, dim=0, index=edge_index[0])
        dst_x = torch.index_select(node_feat, dim=0, index=edge_index[1])
        return self.edge_feat_agger(torch.cat([src_x, dst_x], dim=-1))

    def predict_edge(
        node_feat, activate_nodes,
        num_nodes=None, edge_feat=None
    ):
        if self.sparse and num_nodes is None:
            raise NotImplementedError(
                'sparse backbone requires number of each graph '
                'to obtain correct edge features'
            )
        if not self.sparse and edge_feat is None:
            raise NotImplementedError(
                'dense backbone calculated every pair of edge_features '
                'and the edge features should be provided'
            )

        if self.sparse:
            src_idx, dst_idx, base = [], [], 0
            for idx, p in enumerate(activate_nodes):
                for x, y in combinations(p, 2):
                    src_idx.append((x + base, y + base))
                    dst_idx.append((y + base, x + base))
                base += num_nodes[idx]
            src_idx = torch.LongTensor(src_idx)
            dst_idx = torch.LongTensor(dst_idx)

            ed_feat = self.get_edge_feat(node_feat, src_idx) + \
                self.get_edge_feat(node_feat, dst_idx)

        else:
            ed_feat = []
            for idx, p in enumerate(activate_nodes):
                tx, ty = [], []
                for x, y in combinations(p, 2):
                    tx.append(x)
                    ty.append(y)
                ed_feat.append(edge_feat[idx][tx, ty] + edge_feat[idx][ty, tx])
            ed_feat = torch.cat(ed_feat, dim=0)

        return self.edge_predictor(ed_feat)

    def get_labels(self, activate_nodes, edge_type, empty_type=0):
        edge_feats = []
        for idx, p in enumerate(edge_type):
            for x, y in combinations(p, 2):
                if (x, y) in edge_type[idx]:
                    t_type = edge_type[idx][(x, y)]
                elif (y, x) in edge_type[idx]:
                    t_type = edge_type[idx][(y, x)]
                else:
                    t_type = empty_type
                edge_feats.append(t_type)
        return torch.LongTensor(edge_feats)
