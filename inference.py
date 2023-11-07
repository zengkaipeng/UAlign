from rdkit import Chem
import rdkit
from torch.utils.data import DataLoader
import torch
import os

from Mix_backbone import MixFormer
from Dataset import edit_col_fn
from model import BinaryGraphEditModel, convert_graphs_into_decoder
from data_utils import create_edit_dataset
from utils.chemistry_parse import convert_res_into_smiles


def make_format_logits(node_logits, edge_index, useful_mask, edge_logits):
    node_logits = node_logits.sigmoid()
    edge_logits = edge_logits.sigmoid()
    node_result = {}
    for idx, prob in enumerate(node_logits):
        node_result[idx] = prob.item()

    edge_result = {}
    used_edge = edge_index[:, useful_mask]
    for idx, prob in enumerate(edge_logits):
        src, dst = used_edge[:, idx]
        src, dst = src.item(), dst.item()
        if (src, dst) not in edge_result:
            edge_result[(src, dst)] = edge_result[(dst, src)] = prob.item()
        else:
            t_res = (edge_result[(src, dst)] + prob.item()) / 2
            edge_result[(src, dst)] = edge_result[(dst, src)] = t_res
    return node_result, edge_result


if __name__ == '__main__':

    GNN_TYPE, GATE = 'gat', 'add'
    DIM, HEAD, LAYER = 512, 8, 8
    device = torch.device('cuda:0') if torch.cuda.is_available() \
        else torch.device('cpu')

    if GNN_TYPE == 'gin':
        gnn_args = {'embedding_dim': DIM}
    elif GNN_TYPE == 'gat':
        assert DIM % HEAD == 0, \
            'The model dim should be evenly divided by num_heads'
        gnn_args = {
            'in_channels': DIM, 'out_channels': DIM // HEAD,
            'negative_slope': 0.2, 'dropout': 0.1,
            'add_self_loop': False, 'edge_dim': DIM, 'heads': HEAD
        }
    else:
        raise ValueError(f'Invalid GNN type {GNN_TYPE}')

    GNN = MixFormer(
        emb_dim=DIM, n_layers=LAYER, gnn_args=gnn_args,
        dropout=0.1, heads=HEAD, pos_enc='none',
        negative_slope=0.2, pos_args=None, n_class=None, edge_last=True,
        residual=True, update_gate=GATE, gnn_type=GNN_TYPE
    )

    model = BinaryGraphEditModel(GNN, DIM, DIM, 0.1).to(device)

    rec = [
        '[Br:1][c:2]1[cH:3][cH:4][c:5]([CH:6]=[O:7])[cH:11][cH:12]1.[CH2:8]([CH2:9][OH:10])[OH:13]',
        '[CH2:9]([CH2:10][OH:11])[OH:12].[CH3:1][C:2]([c:3]1[cH:4][cH:5][cH:6][o:7]1)=[O:8]',
        '[NH2:3][c:4]1[cH:5][cH:6][c:7]([N+:8](=[O:9])[O-:10])[cH:11][cH:12]1.[O:1]=[C:2]([C:13]([F:14])([F:15])[F:16])[O:19][C:18](=[O:17])[C:20]([F:21])([F:22])[F:23]',
        '[CH3:1][CH:2]([CH3:3])[C:4](=[O:5])[Cl:10].[NH2:6][C:7]([NH2:8])=[S:9]'
    ]
    prod = [
        '[Br:1][c:2]1[cH:3][cH:4][c:5]([CH:6]2[O:7][CH2:8][CH2:9][O:10]2)[cH:11][cH:12]1',
        '[CH3:1][C:2]1([c:3]2[cH:4][cH:5][cH:6][o:7]2)[O:8][CH2:9][CH2:10][O:11]1',
        '[O:1]=[C:2]([NH:3][c:4]1[cH:5][cH:6][c:7]([N+:8](=[O:9])[O-:10])[cH:11][cH:12]1)[C:13]([F:14])([F:15])[F:16]',
        '[CH3:1][CH:2]([CH3:3])[C:4](=[O:5])[NH:6][C:7]([NH2:8])=[S:9]'
    ]

    dataset = create_edit_dataset(rec, prod, kekulize=False, rxn_class=None)
    col_fn = edit_col_fn(selfloop=GNN_TYPE == 'gat')
    loader = DataLoader(dataset, collate_fn=col_fn, batch_size=2)

    weight_dir = f'../retro_iclr_result/log_edit/wo_class/Gtrans_{GNN_TYPE}'
    weight_name = 'fit-1694174494.1151376.pth'

    model_path = os.path.join(weight_dir, weight_name)

    weight = torch.load(model_path, map_location=device)

    model.load_state_dict(weight)

    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            nod_log, useful_mask, edge_log = model.predict_logitis(data)
            node_res, edge_res = make_format_logits(
                nod_log, data.edge_index, useful_mask, edge_log
            )
            out_graphs = model.predict_into_graphs(data)

            print(node_res, edge_res)
            print(out_graphs)

            result = convert_graphs_into_decoder(out_graphs, 10)
            print(result)

    print('[Making moles]')

    org_node_type = {
        0: 7, 1: 15, 2: 15, 3: 15, 4: 15, 5: 17, 6: 17,
        7: 17, 8: 17, 9: 2, 10: 15, 11: 15
    }
    org_edge_type = {
        (0, 1): 1, (1, 2): 2, (2, 3): 1, (3, 4): 2, (4, 5): 1,
        (7, 8): 1, (8, 9): 1, (4, 10): 1, (10, 11): 2, (11, 1): 1
    }

    pred_node_type = {12: 35}
    pred_edge_type = {(7, 12): 1, (5, 6): 2}

    print(convert_res_into_smiles(
        org_node_type, org_edge_type,
        pred_node_type, pred_edge_type
    ))
