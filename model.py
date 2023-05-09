import torch
from backBone import FCGATEncoder, ExtendedAtomEncoder, ExtendedBondEncoder
from sparse_backBone import (
    GINBase, GATBase, SparseAtomEncoder, SparseBondEncoder
)
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
            torch.nn.Linear(node_dim, 2)
        )
        if self.sparse:
            self.atom_encoder = SparseAtomEncoder(node_dim)
            self.bond_encoder = SparseBondEncoder(edge_dim)
        else:
            self.atom_encoder = ExtendedAtomEncoder(node_dim)
            self.edge_encoder = ExtendedBondEncoder(edge_dim)

    def get_edge_feat(self, node_feat, edge_index):
        assert self.sparse, 'Only sparse mode have edge_feat_agger'
        src_x = torch.index_select(node_feat, dim=0, index=edge_index[:, 0])
        dst_x = torch.index_select(node_feat, dim=0, index=edge_index[:, 1])
        return self.edge_feat_agger(torch.cat([src_x, dst_x], dim=-1))

    def predict_edge(
        self, node_feat, activate_nodes,
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
                # print('[Anodes]', p)
                for x, y in combinations(p, 2):
                    src_idx.append((x + base, y + base))
                    dst_idx.append((y + base, x + base))
                base += num_nodes[idx]

            if len(src_idx) == 0:
                return None

            src_idx = torch.LongTensor(src_idx)
            dst_idx = torch.LongTensor(dst_idx)

            ed_feat = self.get_edge_feat(node_feat, src_idx) + \
                self.get_edge_feat(node_feat, dst_idx)
            print(ed_feat.shape)

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

    def get_init_feats(
        self, graphs, num_nodes=None, num_edges=None, rxn_class=None
    ):
        if self.sparse:
            node_feat = self.atom_encoder(graphs.x, num_nodes, rxn_class)
            edge_feat = self.bond_encoder(
                graphs.edge_attr, num_edges, rxn_class
            )
        else:
            num_nodes = [x['num_nodes'] for x in graphs]
            node_feat = [x['node_feat'] for x in graphs]
            edge_feat = [x['edge_feat'] for x in graphs]
            edge_index = [x['edge_index'] for x in graphs]
            node_feat = self.atom_encoder(num_nodes, node_feat, rxn_class)
            edge_feat = self.bond_encoder(
                num_nodes=num_nodes, edge_index=edge_index,
                edge_feat=edge_feat, rxn_class=rxn_class
            )
        return node_feat, edge_feat

    def update_act_nodes(self, node_res, act_x=None, num_nodes=None):
        node_res = node_res.detach().cpu()
        node_res = torch.argmax(node_res, dim=-1)
        result = []
        if self.sparse:
            base = 0
            for idx, p in enumerate(num_nodes):
                node_res_t = node_res[base: base + p]
                node_all = torch.arange(p)
                mask = node_res_t == 1
                t_result = set(node_all[mask].tolist())
                if act_x is not None:
                    t_result |= set(act_x[idx])
                result.append(list(t_result))
                base += p
        return result

    def forward(
        self, graphs, act_nodes=None, num_nodes=None, num_edges=None,
        attn_mask=None, rxn_class=None, mode='together', return_feat=False
    ):
        node_feat, edge_feat = self.get_init_feats(
            graphs, num_nodes, num_edges, rxn_class
        )
        if self.sparse:
            node_feat, _ = self.base_model(
                node_feats=node_feat, edge_feats=edge_feat,
                edge_index=graphs.edge_index
            )
            edge_feat, node_res = None, self.node_predictor(node_feat)

        if mode in ['together', 'inference']:
            act_nodes = self.update_act_nodes(
                act_x=act_nodes if mode == 'together' else None,
                node_res=node_res, num_nodes=num_nodes
            )
        elif mode != 'original':
            raise NotImplementedError(f'Invalid mode: {mode}')

        pred_edge = self.predict_edge(
            node_feat, act_nodes, num_nodes, edge_feat
        )
        if return_feat:
            return node_res, pred_edge, act_nodes, node_feat, edge_feat
        else:
            return node_res, pred_edge, act_nodes


def get_labels(activate_nodes, edge_type, empty_type=0):
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


def evaluate_sparse(
    node_res, pred_edge, num_nodes, num_edges,
    edge_types, act_nodes, used_nodes
):
    base, e_base, total = 0, 0, 0
    node_cover, node_fit, edge_fit, all_fit, all_cover = 0, 0, 0, 0, 0
    node_res = node_res.cpu().argmax(dim=-1)
    edge_res = edge_res.cpu().argmax(dim=-1)
    for idx, p in enumerate(num_nodes):
        t_node_res = node_res[base: base + p] > 0.5
        real_nodes = torch.zeros_like(t_node_res, dtype=bool)
        real_nodes[act_nodes[idx]] = True
        inters = torch.logical_and(real_nodes, t_node_res)
        nf = torch.all(real_nodes == t_node_res).item()
        nc = torch.all(read_nodes == inters).item()

        node_all = torch.arange(p)

        edge_labels = get_labels(used_nodes[idx], edge_types[idx])
        e_size = len(edge_labels)

        ef = torch.all(edge_labels == edge_res[e_base: e_base + e_size]).item()

        base, e_base = base + p, e_base + e_size
        total += 1
        node_fit += nf
        node_cover += nc
        edge_fit += ef
        all_fit += (nf & ef)
        all_cover += (nc & ef)
    return node_cover, node_fit, edge_fit, all_fit, all_cover, total
