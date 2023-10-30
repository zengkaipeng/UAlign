import torch
from sparse_backBone import (
    GINBase, GATBase, SparseAtomEncoder, SparseBondEncoder
)
from itertools import combinations, permutations
from torch_geometric.data import Data
from typing import Any, Dict, List, Tuple, Optional, Union
import numpy as np
from torch.nn.functional import binary_cross_entropy_with_logits
from torch.nn.functional import cross_entropy
from scipy.optimize import linear_sum_assignment


class BinaryGraphEditModel(torch.nn.Module):
    def __init__(self, base_model, node_dim, edge_dim, dropout=0.1):
        super(BinaryGraphEditModel, self).__init__()
        self.base_model = base_model
        self.edge_predictor = torch.nn.Sequential(
            torch.nn.Linear(edge_dim, edge_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(edge_dim, 1)
        )

        self.node_predictor = torch.nn.Sequential(
            torch.nn.Linear(node_dim, node_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(node_dim, 1)
        )

    def calc_loss(
        self, node_logits, node_label, edge_logits, edge_label,
        node_batch, edge_batch, pos_weight=1
    ):
        max_node_batch = node_batch.max().item() + 1
        node_loss = torch.zeros(max_node_batch).to(node_logits)
        node_loss_src = binary_cross_entropy_with_logits(
            node_logits, node_label, reduction='none',
            pos_weight=torch.tensor(pos_weight)
        )
        node_loss.scatter_add_(0, node_batch, node_loss_src)

        max_edge_batch = edge_batch.max().item() + 1
        edge_loss = torch.zeros(max_edge_batch).to(edge_logits)
        edge_loss_src = binary_cross_entropy_with_logits(
            edge_logits, edge_label, reduction='none',
            pos_weight=torch.tensor(pos_weight)
        )
        edge_loss.scatter_add_(0, edge_batch, edge_loss_src)

        return node_loss.mean(), edge_loss.mean()

    def forward(self, graph, ret_loss=True, ret_feat=False, pos_weight=1):
        node_feat, edge_feat = self.base_model(graph)

        node_logits = self.node_predictor(node_feat)
        node_logits = node_logits.squeeze(dim=-1)

        edge_logits = self.edge_predictor(edge_feat)
        edge_logits = edge_logits.squeeze(dim=-1)

        if ret_loss:
            n_loss, e_loss = self.calc_loss(
                node_logits=node_logits, edge_logits=edge_logits,
                node_label=graph.node_label, node_batch=graph.batch,
                edge_label=graph.edge_label, edge_batch=graph.e_batch,
                pos_weight=pos_weight
            )

        answer = (node_logits, edge_logits)
        if ret_loss:
            answer += (n_loss, e_loss)
        if ret_feat:
            answer += (node_feat, edge_feat)
        return answer

    def make_memory(self, graph):
        node_feat, edge_feat = self.base_model(graph)
        batch_size, max_node = graph.batch_mask.shape

        memory = torch.zeros(batch_size, max_node, node_feat.shape[-1])
        memory = memory.to(device)
        memory[graph.batch_mask] = node_feat

        return memory, graph.batch_mask

    def predict_into_graphs(self, graph):
        node_feat, edge_feat = self.base_model(graph)
        node_logits = self.node_predictor(node_feat).squeeze(dim=-1)
        node_pred = node_logits.detach().clone()
        node_pred[node_pred > 0] = 1
        node_pred[node_pred <= 0] = 0
        useful_mask = self.make_useful_mask(
            graph.edge_index, mode='inference',
            pred_label=node_pred
        )

        device = node_feat.device

        edge_res = {}
        if torch.any(useful_mask):
            edge_logits = self.edge_predictor(edge_feat[useful_mask])
            edge_logits = edge_logits.squeeze(-1).sigmoid().tolist()
            used_edges = graph.edge_index[:, useful_mask]

            for idx, res in enumerate(edge_logits):
                src, dst = used_edges[:, idx]
                src, dst = src.item(), dst.item()
                if src == dst:
                    continue
                if (src, dst) not in edge_res:
                    edge_res[(src, dst)] = edge_res[(dst, src)] = res
                else:
                    real_log = (edge_res[(src, dst)] + res) / 2
                    edge_res[(src, dst)] = edge_res[(dst, src)] = real_log

        splited_graph = self.seperate_a_graph(graph)
        all_node_index = torch.arange(0, node_feat.shape[0]).to(device)

        meta_graphs = []
        for idx, org_graph in enumerate(splited_graph):
            this_node_mask = graph.batch == idx

            res_edge, res_feat, offset = [], [], org_graph['offset']
            for edx, eattr in enumerate(org_graph['edge_attr']):
                src = org_graph['edge_index'][0, edx].item()
                dst = org_graph['edge_index'][1, edx].item()
                if edge_res.get((src, dst), 0) < 0.5:
                    res_edge.append((src - offset, dst - offset))
                    res_feat.append(eattr)

            # activate nodes
            activate_nodes = all_node_index[(node_logits > 0) & this_node_mask]
            activate_nodes = (activate_nodes - offset).tolist()

            meta_graph = {
                'x': org_graph['x'], 'act_node': activate_nodes,
                'self_edge': org_graph['self_edge'] - offset,
                'res_edge': torch.LongTensor(res_edge).T,
                'edge_attr': torch.stack(res_feat, dim=0),
                'num_nodes': org_graph['num_nodes']
            }
            if 'rxn' in org_graph:
                meta_graph['rxn'] = org_graph['rxn']
            meta_graphs.append(meta_graph)
        return meta_graphs


def make_diag_by_mask(max_node, mask):
    x = torch.zeros(max_node, max_node)
    x[mask] = 1
    x[:, ~mask] = 0
    return x.bool()


def convert_graphs_into_decoder(graphs, pad_num):

    def dfs(x, graph, blocks, vis):
        blocks.append(x)
        vis.add(x)
        for neighbor in graph[x]:
            if neighbor not in vis:
                dfs(neighbor, graph, blocks, vis)

    def make_block(edge_index, max_node, pad_idx):
        # print('pad_idx', max_node, pad_idx)
        attn_mask = torch.zeros(max_node, max_node).bool()
        graph, vis = {}, set()
        for idx in range(edge_index.shape[1]):
            row, col = edge_index[:, idx].tolist()
            if row not in graph:
                graph[row] = []
            graph[row].append(col)

        for idx in range(max_node):
            if idx not in graph and idx not in pad_idx:
                graph[idx] = [idx]

        for node in graph.keys():
            if node in vis:
                continue
            block = []
            dfs(node, graph, block, vis)
            x_mask = torch.zeros(max_node).bool()
            # print('block', block)
            x_mask[block] = True
            x_mask[pad_idx] = True
            # print('x_mask', x_mask)
            block_attn = make_diag_by_mask(max_node, x_mask)
            # print('block_attn', block_attn.long())
            attn_mask |= block_attn
        return attn_mask

    node_feat, edge_feat, edge_index = [], [], []
    self_mask, org_mask, pad_mask = [], [], []
    node_org_mask, node_pad_mask, attn_mask = [], [], []
    node_rxn, edge_rxn, graph_rxn = [], [], []
    node_batch, node_ptr, edge_batch, edge_ptr = [], [0], [], [0]
    lst_node, lst_edge = 0, 0

    max_node = max(x['num_nodes'] + pad_num for x in graphs)

    for idx, graph in enumerate(graphs):
        node_feat.append(graph['x'])
        node_pad_mask.append(torch.zeros(graph['num_nodes']).bool())
        node_org_mask.append(torch.ones(graph['num_nodes']).bool())
        node_org_mask.append(torch.zeros(pad_num).bool())
        node_pad_mask.append(torch.ones(pad_num).bool())

        edge_feat.append(graph['edge_attr'])
        attn_mask.append(make_block(
            edge_index=graph['res_edge'], max_node=max_node,
            pad_idx=[x + graph['num_nodes'] for x in range(pad_num)]
        ))

        # org_edge
        edge_index.append(graph['res_edge'] + lst_node)
        res_num = graph['res_edge'].shape[1]
        self_mask.append(torch.zeros(res_num).bool())
        org_mask.append(torch.ones(res_num).bool())
        pad_mask.append(torch.zeros(res_num).bool())

        exist_edges = set(
            (row.item(), col.item()) for row, col in
            graph['res_edge'].T
        )

        # print('[res_num]', res_num)

        # self_edge

        edge_index.append(graph['self_edge'] + lst_node)
        self_num = graph['self_edge'].shape[1]
        self_mask.append(torch.ones(self_num).bool())
        org_mask.append(torch.zeros(self_num).bool())
        pad_mask.append(torch.zeros(self_num).bool())

        link_idx = [x + graph['num_nodes'] for x in range(pad_num)]
        link_idx.extend(graph['act_node'])

        pad_edges = [
            (x, y) for x in link_idx for y in link_idx
            if x != y and (x, y) not in exist_edges
        ]
        pad_e_num = len(pad_edges)
        pad_edges = torch.LongTensor(pad_edges).T + lst_node
        edge_index.append(pad_edges)

        self_mask.append(torch.zeros(pad_e_num).bool())
        org_mask.append(torch.zeros(pad_e_num).bool())
        pad_mask.append(torch.ones(pad_e_num).bool())

        node_batch.append(
            torch.ones(graph['num_nodes'] + pad_num).long() * idx
        )
        edge_batch.append(
            torch.ones(self_num + res_num + pad_e_num).long() * idx
        )

        lst_node += graph['num_nodes'] + pad_num
        lst_edge += self_num + res_num + pad_e_num

        node_ptr.append(lst_node)
        edge_ptr.append(lst_edge)

        if 'rxn' in graph:
            rxn = graph['rxn']
            node_rxn.append(torch.ones(graph['num_nodes']).long() * rxn)
            edge_rxn.append(torch.ones(self_num + res_num).long() * rxn)
            graph_rxn.append(rxn)

    result = {
        'x': torch.cat(node_feat, dim=0),
        'edge_attr': torch.cat(edge_feat, dim=0),
        'edge_index': torch.cat(edge_index, dim=1),
        'num_nodes': lst_node, 'num_edges': lst_edge,
        'batch': torch.cat(node_batch, dim=0),
        'e_batch': torch.cat(edge_batch, dim=0),
        'ptr': torch.LongTensor(node_ptr),
        'e_ptr': torch.LongTensor(edge_ptr),
        'org_mask': torch.cat(org_mask, dim=0),
        'self_mask': torch.cat(self_mask, dim=0),
        "pad_mask": torch.cat(pad_mask, dim=0),
        'node_org_mask': torch.cat(node_org_mask, dim=0),
        'node_pad_mask': torch.cat(node_pad_mask, dim=0),
        'attn_mask': torch.stack(attn_mask, dim=0)
    }
    if len(graph_rxn) > 0:
        result['graph_rxn'] = torch.LongTensor(graph_rxn)
        result['node_rxn'] = torch.cat(node_rxn, dim=0)
        result['edge_rxn'] = torch.cat(edge_rxn, dim=0)
    return Data(**result)


def evaluate_sparse(
    node_pred, edge_pred, node_batch, edge_batch,
    node_label, edge_label, batch_size
):
    node_cover, node_fit, edge_fit, edge_cover, all_fit, all_cover = [0] * 6
    node_ptr = make_ptr_from_batch(node_batch, batch_size)
    if edge_batch.numel() > 0:
        edge_ptr = make_ptr_from_batch(edge_batch, batch_size)
    for idx in range(batch_size):
        t_node_res = node_pred[node_ptr[idx]: node_ptr[idx + 1]] > 0
        real_nodes = node_label[node_ptr[idx]: node_ptr[idx + 1]] > 0
        inters = torch.logical_and(t_node_res, real_nodes)
        nf = torch.all(real_nodes == t_node_res).item()
        nc = torch.all(real_nodes == inters).item()

        if edge_batch.numel() > 0:
            t_edge_res = edge_pred[edge_ptr[idx]: edge_ptr[idx + 1]] > 0
            real_edges = edge_label[edge_ptr[idx]: edge_ptr[idx + 1]] > 0
            e_inters = torch.logical_and(t_edge_res, real_edges)
            ef = torch.all(t_edge_res == real_edges).item()
            ec = torch.all(real_edges == e_inters).item()
        else:
            ef = ec = True

        node_fit += nf
        node_cover += nc
        edge_fit += ef
        edge_cover += ec
        all_fit += (nf & ef)
        all_cover += (nc & ec)
    return node_cover, node_fit, edge_cover, edge_fit, all_cover, all_fit


class DecoderOnly(torch.nn.Module):
    def __init__(
        self, backbone, node_dim, edge_dim,
        node_class, edge_class
    ):
        super(DecoderOnly, self).__init__()
        self.backbone = backbone
        self.node_predictor = torch.nn.Linear(node_dim, node_class)
        self.edge_predictor = torch.nn.Linear(edge_dim, edge_class)

    def forward(self, graph, memory, mem_pad_mask=None):
        node_feat, edge_feat = self.backbone(graph, memory, mem_pad_mask)
        node_pred = self.node_predictor(node_feat)
        edge_pred = self.edge_predictor(edge_feat)
        return node_pred, edge_pred

    def predict_paddings(self, graph, memory, mem_pad_mask=None):
        node_feat, edge_feat = self.backbone(graph, memory, mem_pad_mask)
        node_pred = self.node_predictor(node_feat)
        edge_pred = self.edge_predictor(edge_feat)

        edge_pred = torch.softmax(edge_pred, dim=-1)
        node_pred = node_pred.argmax(dim=-1)

        batch_size = graph.batch.max().item() + 1
        device = graph.x.device

        all_node_index = torch.arange(0, graph.num_nodes).to(device)
        all_node_res, all_edge_res = [], []
        for i in range(batch_size):
            pad_n_mask = (graph.batch == i) & graph.node_pad_mask
            pad_e_mask = (graph.e_batch == i) & graph.pad_mask
            used_n_mask = (graph.batch == i) & graph.node_org_mask
            offset = graph.ptr[i].item()

            this_node_pred = node_pred[pad_n_mask]
            this_node_idx = all_node_index[pad_n_mask]

            useful_nodes = (all_node_index[used_n_mask] - offset).tolist()
            useful_nodes = set(useful_nodes)

            node_res = {}
            for idx, p in enumerate(this_node_pred):
                if p.item() != 0:
                    n_idx = this_node_idx[idx].item() - offset
                    node_res[n_idx] = p.item()
                    useful_nodes.add(n_idx)

            edge_log = {}

            this_edge_pred = edge_pred[pad_e_mask]
            this_edge_idx = graph.edge_index[:, pad_e_mask]

            # print(this_edge_idx.T)

            for idx, logs in enumerate(this_edge_pred):
                row, col = this_edge_idx[:, idx]
                row, col = row.item(), col.item()
                if row not in useful_nodes or col not in useful_nodes:
                    continue
                if (row, col) not in edge_log:
                    edge_log[(row, col)] = edge_log[(col, row)] = logs
                else:
                    real_logs = (logs + edge_log[(row, col)]) / 2
                    edge_log[(row, col)] = edge_log[(col, row)] = real_logs

            edge_res = {}
            for (x, y), v in edge_log.items():
                if (x, y) not in edge_res and (y, x) not in edge_res:
                    pred_value = v.argmax().item()
                    if pred_value != 0:
                        edge_res[(x, y)] = pred_value

            all_node_res.append(node_res)
            all_edge_res.append(edge_res)

        return all_node_res, all_edge_res


class EncoderDecoder(torch.nn.Module):
    def __init__(self, encoder, decoder):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(
        self, encoder_graph, decoder_graph, encoder_mode, edge_types,
        reduction='mean', graph_level=True, alpha=1
    ):
        encoder_answer = self.encoder(
            graph=encoder_graph, graph_level=graph_level,
            mask_mode=encoder_mode, ret_loss=True,
            ret_feat=True, reduce_mode=reduction,
        )

        node_pred, edge_pred, useful_mask, n_loss, e_loss, \
            node_feat, edge_feat = encoder_answer

        device = encoder_graph.x.device
        batch_size = encoder_graph.batch.max().item() + 1
        n_nodes = torch.zeros(batch_size).long().to(device)
        n_nodes.scatter_add_(
            src=torch.ones_like(encoder_graph.batch),
            dim=0, index=encoder_graph.batch
        )
        max_node = n_nodes.max().item() + 1
        mem_pad_mask = make_batch_mask(encoder_graph.ptr, max_node, batch_size)

        memory = torch.zeros(batch_size, max_node, node_feat.shape[-1])
        memory = memory.to(device)
        memory[mem_pad_mask] = node_feat

        node_logits, edge_logits = self.decoder(
            graph=decoder_graph, memory=memory,
            mem_pad_mask=mem_pad_mask
        )

        decoder_losses = self.loss_calc(
            graph=decoder_graph, node_logits=node_logits,
            reduction=reduction, edge_logits=edge_logits,
            edge_type_dict=edge_types, graph_level=graph_level
        )
        org_node_loss, org_edge_loss, x_loss, c_loss = decoder_losses

        return n_loss + e_loss + x_loss + c_loss \
            + (org_node_loss + org_edge_loss) * alpha

    def get_matching(self, node_logits, pad_node_label):
        # print(node_logits.shape)
        neg_node_log_prob = -torch.log_softmax(node_logits, dim=-1)
        # print(neg_node_log_prob.shape)
        val_matrix = neg_node_log_prob[:, pad_node_label]
        val_matrix = val_matrix.cpu().numpy()
        row_id, col_id = linear_sum_assignment(val_matrix)
        return row_id, col_id

    def loss_calc(
        self, graph, node_logits, edge_logits,
        edge_type_dict, reduction='mean', graph_level=True,
    ):
        """[summary]

        calc the loss for decoder with bin matching

        Args:
            node_logits ([type]): A tensor of (N nodes, n_class), the prediction
                for the whole batch nodes
            edge_logits ([type]): A tensor of (N edge, e_class), the prediction
                for the whole batch edge
            org_node_label ([type]): A tensor of (N node_org), the label 
                of all the original nodes
            org_edge_label ([type]): A tensor of (N edge_org), the label for 
                all the original edges 
            pad_node_label ([type]): A tesnor of (batch_size, N_pad), the label
                for all the padding nodes
            edge_type_dict ([type]): A dict containing all the edge_type of 
                every edges in the batch, in form of (idx_i, idx_j) -> e_type
            reduce (str): the reduction method of loss (default: `'mean'`)
            graph_level (bool): reduce all the losses first to graph 
                then to the other, (default: `True`)
            alpha (number): the weight of org part, as aux loss (default: `1`)
        """
        batch_size = graph.batch.max().item() + 1
        device = node_logits.device
        org_node_logits = node_logits[graph.node_org_mask]

        org_edge_logits = edge_logits[graph.org_mask]
        org_node_labels = graph.node_class[graph.node_org_mask]
        pad_node_labels = graph.node_class[graph.node_pad_mask]
        pad_node_labels = pad_node_labels.reshape(batch_size, -1)

        if graph_level:
            org_node_loss = cross_entropy(
                org_node_logits, org_node_labels, reduction='none'
            )
            org_edge_loss = cross_entropy(
                org_edge_logits, graph.org_edge_class, reduction='none'
            )
            org_node_batch = graph.batch[graph.node_org_mask]
            org_edge_batch = graph.e_batch[graph.org_mask]
            org_node_loss_graph = torch.zeros(batch_size).to(device)
            org_edge_loss_graph = torch.zeros(batch_size).to(device)

            org_node_loss_graph.scatter_add_(
                dim=0, src=org_node_loss, index=org_node_batch
            )
            org_edge_loss_graph.scatter_add_(
                dim=0, src=org_edge_loss, index=org_edge_batch
            )

            if reduction == 'mean':
                org_node_loss = org_node_loss_graph.mean()
                org_edge_loss = org_edge_loss_graph.mean()
            else:
                org_node_loss = org_node_loss_graph.sum()
                org_edge_loss = org_edge_loss_graph.sum()

        else:
            org_node_loss = cross_entropy(
                org_node_logits, org_node_labels, reduction=reduction
            )
            org_edge_loss = cross_entropy(
                org_edge_logits, graph.org_edge_class, reduction=reduction
            )

        # get loss by matching

        if graph_level:
            total_x_loss = torch.zeros(batch_size).to(device)
            total_e_loss = torch.zeros(batch_size).to(device)
        else:
            total_e_loss, total_x_loss = [], []

        all_node_index = torch.arange(graph.num_nodes).to(device)
        pad_index = all_node_index[graph.node_pad_mask].reshape(batch_size, -1)
        pad_result = node_logits[graph.node_pad_mask].reshape(
            batch_size, pad_index.shape[1], -1
        )

        for idx, n_lg in enumerate(pad_result):
            this_node_label = pad_node_labels[idx]
            this_pad_idx = pad_index[idx]
            with torch.no_grad():
                row_id, col_id = self.get_matching(n_lg, this_node_label)
                # row our col gt

            node_pad = row_id.shape[0]
            node_reidx = torch.zeros(node_pad).long()
            node_reidx[row_id.tolist()] = torch.from_numpy(col_id)
            node_reidx = node_reidx.to(device)

            if graph_level:
                x_loss = cross_entropy(
                    n_lg, this_node_label[node_reidx],
                    reduction='sum'
                )
                total_x_loss[idx] = x_loss
            else:
                x_loss = cross_entropy(
                    n_lg, this_node_label[node_reidx],
                    reduction='none'
                )
                total_x_loss.append(x_loss)

            node_remap = {
                this_pad_idx[row].item(): this_pad_idx[col_id[tx]].item()
                for tx, row in enumerate(row_id)
            }
            # print(node_remap)
            used_node_set = set(
                this_pad_idx[idx].item() for idx, lb in
                enumerate(this_node_label) if lb != 0
            )
            this_org_node_mask = torch.logical_and(
                graph.batch == idx, graph.node_org_mask
            )
            used_node_set.update(all_node_index[this_org_node_mask].tolist())

            this_pad_mask = torch.logical_and(
                graph.e_batch == idx, graph.pad_mask
            )

            # print(used_node_set)

            this_edge_idx = graph.edge_index[:, this_pad_mask]
            used_edge_label, used_edge_log = [], []
            this_edge_log = edge_logits[this_pad_mask]

            for ex in range(this_edge_idx.shape[1]):
                idx_i = this_edge_idx[0][ex].item()
                idx_j = this_edge_idx[1][ex].item()
                idx_i = node_remap.get(idx_i, idx_i)
                idx_j = node_remap.get(idx_j, idx_j)
                if idx_i not in used_node_set or idx_j not in used_node_set:
                    continue
                used_edge_log.append(this_edge_log[ex])
                used_edge_label.append(edge_type_dict.get((idx_i, idx_j), 0))
                # print(idx_i, idx_j)

            used_edge_label = torch.LongTensor(used_edge_label).to(device)
            used_edge_log = torch.stack(used_edge_log, dim=0)

            if graph_level:
                e_loss = cross_entropy(
                    used_edge_log, used_edge_label, reduction='sum'
                )
                total_e_loss[idx] = e_loss

            else:
                e_loss = cross_entropy(
                    used_edge_log, used_edge_label, reduction='none'
                )
                total_e_loss.append(e_loss)

        if not graph_level:
            total_x_loss = torch.cat(total_x_loss, dim=0)
            total_e_loss = torch.cat(total_e_loss, dim=0)

        if reduction == 'sum':
            total_x_loss = total_x_loss.sum()
            total_e_loss = total_e_loss.sum()
        else:
            total_e_loss = total_e_loss.mean()
            total_x_loss = total_x_loss.mean()

        return org_node_loss, org_edge_loss, total_x_loss, total_e_loss
