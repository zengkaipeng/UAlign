import torch
from backBone import FCGATEncoder, ExtendedAtomEncoder, ExtendedBondEncoder
from sparse_backBone import (
    GINBase, GATBase, SparseAtomEncoder, SparseBondEncoder
)
from itertools import combinations, permutations
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


def get_collate_fn(sparse, self_loop):
    def graph_collate_fn(data_batch):
        batch_size, rxn_class, node_label = len(data_batch), [], []
        edge_idxes, edge_feats, node_feats = [], [], []
        batch, lstnode, ptr, edge_map = [], 0, [0], []
        activate_nodes, edge_types = [], []
        node_rxn, edge_rxn, lstedge = [], [], 0
        for idx, data in enumerate(data_batch):
            if len(data) == 4:
                graph, n_lb,  e_type, A_node = data
                r_class = None
            else:
                graph, r_class, n_lb,  e_type, A_node = data
                rxn_class.append(r_class)

            # labels
            edge_types.append(e_type)
            node_label.append(n_lb)
            activate_nodes.append(A_node)

            # graph info

            cnt_node = graphs['num_nodes']
            cnt_edge = graph['edge_index'].shape[1]

            edge_idxes.append(graph['edge_index'] + lstnode)
            edge_feats.append(graph['edge_feat'])
            node_feats.append(graph['node_feat'])
            lstnode += cnt_node
            ptr.append(lstnode)
            batch.append(np.ones(cnt_node, dtype=np.int64) * idx)

            # r_class

            if r_class is not None:
                node_rxn.append(np.ones(cnt_edge, dtype=np.int64) * r_class)
                edge_rnx.append(np.ones(cnt_edge, dtype=np.int64) * r_class)

            # edge_mapping
            e_map = {}
            for tdx in range(cnt_edge):
                x = int(graph['edge_index'][0][tdx])
                y = int(graph['edge_index'][1][tdx])
                e_map[(x, y)] = tdx + lstedge
            lstedge += cnt_edge
            edge_map.append(e_map)

        result, more_rxn = {}, []

        # sparse padding
        if not sparse:
            result['original_edge_ptr'] = lstedge
            pad_edge, self_edge = [], []
            for idx in range(batch_size):
                e_map, cnt_node = {}, ptr[idx + 1] - ptr[idx]
                all_idx = list(range(cnt_node))
                for tdx, (x, y) in enumerate(permutations(all_idx, 2)):
                    if (x, y) not in edge_map[idx]:
                        e_map[(x, y)] = len(pad_edge) + lstedge
                        pad_edge.append((x + ptr[idx], y + ptr[idx]))
                        if len(rxn_class) != 0:
                            more_rxn.append(rxn_class[idx])

                edge_map[idx].update(e_map)

            if len(pad_edge) > 0:
                lstedge += len(pad_edge)
                edge_idxes.append(np.array(pad_edge, dtype=np.int64).T)
        result['pad_edge_ptr'] = lstedge

        # self loops

        if self_loop:
            for idx in range(batch_size):
                cnt_node = ptr[idx + 1] - ptr[idx]
                for tdx in range(cnt_node):
                    if (x, x) not in edge_map[idx]:
                        edge_map[idx][(x, x)] = len(self_edge) + lstedge
                        self_edge.append((x + ptr[idx], x + ptr[idx]))
                        if len(rxn_class) != 0:
                            more_rxn.append(rxn_class[idx])
            if len(self_edge) > 0:
                lstedge += len(self_edge)
                edge_idxes.append(np.array(self_edge, dtype=np.int64).T)
        result['self_edge_ptr'] = lstedge

        # result merging

        result['edge_index'] = torch.from_numpy(
            np.concatenate(edge_idxes, axis=-1)
        )
        result['x'] = torch.from_numpy(np.concatenate(node_feats, axis=0))
        result['edge_attr'] = torch.from_numpy(
            np.concatenate(edge_feats, axis=0)
        )
        result['ptr'] = torch.LongTensor(ptr)
        result['batch'] = torch.from_numpy(np.concatenate(batch, axis=0))
        if len(rxn_class) == 0:
            return Data(**result), node_label, edge_types, \
                activate_nodes, edge_map
        else:
            result['rxn_node'] = torch.from_numpy(
                np.concatenate(node_rxn, axis=0)
            )
            result['rxn_edge'] = torch.from_numpy(
                np.concatenate(edge_rxn, axis=0)
            )
            if len(more_rxn) > 0:
                result['rxn_edge'] = torch.cat([
                    result['rxn_edge'], torch.LongTensor(more_rxn)
                ], dim=0)
            return Data(**result), rxn_class, node_label, edge_types,\
                activate_nodes, edge_map

    return graph_collate_fn


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
        )o: Any, name: str

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
        self, node_feat, activate_nodes, edge_feat=None,
        e_types=None, edge_map=None, empty_type=0, ptr=None
    ):
        if self.sparse and ptr is None:
            raise NotImplementedError(
                'sparse backbone requires number of each graph '
                'to obtain correct edge features'
            )
        if not self.sparse and (edge_feat is None or edge_map is None):
            raise NotImplementedError(
                'dense backbone calculated every pair of edge_features '
                'and the edge features should be provided'
            )
        e_answer, e_ptr = [], [0]

        if self.sparse:
            src_idx, dst_idx = [], []
            for idx, p in enumerate(activate_nodes):
                # print('[Anodes]', p)
                for x, y in combinations(p, 2):
                    src_idx.append((x + ptr[idx], y + ptr[idx]))
                    dst_idx.append((y + ptr[idx], x + ptr[idx]))
                    e_answer.append(get_label(e_types[idx], x, y, empty_type))
                e_ptr.append(len(e_answer))

            if len(src_idx) == 0:
                return None, [], [0]

            src_idx = torch.LongTensor(src_idx)
            dst_idx = torch.LongTensor(dst_idx)

            ed_feat = self.get_edge_feat(node_feat, src_idx) + \
                self.get_edge_feat(node_feat, dst_idx)

        else:
            src_idx, dst_idx = [], []
            for idx, p in enumerate(activate_nodes):
                for x, y in combinations(p, 2):
                    src_idx.append(edge_map[(x, y)])
                    dst_idx.append(edge_map[(y, x)])
                    e_answer.append(get_label(e_types[idx], x, y, empty_type))
                e_ptr.append(len(e_answer))

            if len(src_idx) == 0:
                return None, [], [0]

            ed_feat = edge_feat[src_idx] + edge_feat[dst_idx]

        return self.edge_predictor(ed_feat), e_answer, e_ptr

    def get_init_feats(self, graphs):
        rxn_node = getattr(graphs, 'rxn_node', None)
        rxn_edge = getattr(graphs, 'rxn_edge', None)
        node_feat = self.atom_encoder(node_feat=graphs.x, rxn_class=rxn_node)
        edge_feat = self.bond_encoder(
            edge_feat=graphs.edge_attr, org_ptr=graphs.original_edge_ptr,
            pad_ptr=graphs.pad_edge_ptr, self_ptr=graphs.self_edge_ptr,
            rxn_class=rxn_edge
        )
        return node_feat, edge_feat

    def update_act_nodes(self, node_res, ptr, act_x=None):
        node_res = node_res.detach().cpu()
        node_res = torch.argmax(node_res, dim=-1)
        result = []
        for idx in range(len(ptr) - 1):
            node_res_t = node_res[ptr[idx]: ptr[idx + 1]]
            node_all = torch.arange(ptr[idx + 1] - ptr[idx])
            mask = node_res_t == 1
            t_result = set(node_all[mask].tolist())
            if act_x is not None:
                t_result = set(act_x[idx])
            result.append(t_result)
        return result

    def forward(
        self, graphs, act_nodes=None, mode='together', e_types=None,
        empty_type=0, edge_map=None
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

        pred_edge, e_answer, e_ptr = self.predict_edge(
            node_feat=node_feat, activate_nodes=act_nodes, edge_faet=edge_feat,
            e_types=e_types, empty_type=empty_type, edge_map=edge_map,
            ptr=graphs.ptr
        )
        return node_res, pred_edge, e_answer, e_ptr, act_nodes


def get_label(e_type, x, y, empty_type=0):
    if e_type is None:
        return empty_type
    if (x, y) in e_type:
        return e_type[(x, y)]
    elif (y, x) in e_type:
        return e_type[(y, x)]
    else:
        return empty_type


def evaluate_sparse(node_res, pred_edge, e_labels, node_ptr, e_ptr, act_nodes):
    node_cover, node_fit, edge_fit, all_fit, all_cover = 0, 0, 0, 0, 0
    node_res = node_res.cpu().argmax(dim=-1)
    if edge_res is not None:
        edge_res = edge_res.cpu().argmax(dim=-1)
    for idx, a_node in enumerate(act_nodes):
        t_node_res = node_res[node_ptr[idx]: node_ptr[idx] + 1] == 1
        real_nodes = torch.zeros_like(t_node_res, dtype=bool)
        real_nodes[a_node] = True
        inters = torch.logical_and(real_nodes, t_node_res)
        nf = torch.all(real_nodes == t_node_res).item()
        nc = torch.all(read_nodes == inters).item()

        t_edge_res = edge_res[e_ptr[idx]: e_ptr[idx + 1]]
        t_edge_labels = e_labels[e_ptr[idx]: e_ptr[idx + 1]]
        ef = torch.all(t_edge_res == t_edge_labels).item()

        node_fit += nf
        node_cover += nc
        edge_fit += ef
        all_fit += (nf & ef)
        all_cover += (nc & ef)
    return node_cover, node_fit, edge_fit, all_fit, all_cover, len(act_nodes)
