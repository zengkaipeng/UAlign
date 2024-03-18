import os
import argparse
import pickle
from sparse_backBone import GATBase
from model import PositionalEncoding, PretrainModel
import torch
from torch.nn import TransformerDecoder, TransformerDecoderLayer
from utils.chemistry_parse import cano_with_am, find_all_amap, remove_am_wo_cano
from utils.graph_utils import smiles2graph
from tokenlizer import smi_tokenizer
from rdkit import Chem
import torch_geometric
from data_utils import generate_tgt_mask


def make_graph_batch(smi, rxn=None):
    graph = smiles2graph(smi, with_amap=False)
    num_nodes = graph['node_feat'].shape[0]
    num_edges = graph['edge_index'].shape[1]

    data = {
        'x': torch.from_numpy(graph['node_feat']),
        'num_nodes': num_nodes,
        'edge_attr': torch.from_numpy(graph['edge_feat']),
        'edge_index': torch.from_numpy(graph['edge_index']),
        'ptr': torch.LongTensor([0, num_nodes]),
        'e_ptr': torch.LongTensor([0, num_edges]),
        'batch': torch.zeros(num_nodes).long(),
        'e_batch': torch.zeros(num_edges).long(),
        'batch_mask': torch.ones(1, num_nodes).bool()
    }

    if rxn is not None:
        data['node_rxn'] = torch.ones(num_nodes).long() * rxn
        data['edge_rxn'] = torch.ones(num_edges).long() * rxn
    return torch_geometric.data.Data(**data)


def process_reac_via_prod(prod, reac):
    pro_atom_maps = find_all_amap(prod)
    reacts = reac.split('.')
    rea_atom_maps = [find_all_amap(x) for x in reacts]

    aligned_reactants = []
    for i, rea_map_num in enumerate(rea_atom_maps):
        for j, mapnum in enumerate(pro_atom_maps):
            if mapnum in rea_map_num:
                mol = Chem.MolFromSmiles(reacts[i])
                amap = {
                    x.GetAtomMapNum(): x.GetIdx() for x in mol.GetAtoms()
                }

                y_smi = Chem.MolToSmiles(
                    mol, rootedAtAtom=amap[mapnum], canonical=True
                )

                aligned_reactants.append((y_smi, j))
                break

    aligned_reactants.sort(key=lambda x: x[1])
    return '.'.join(x[0] for x in aligned_reactants)


def fwd_hood(module, inputs, outputs, ip_cont, op_cont):
    ip_cont.append(inputs)
    op_cont.append(outputs)


def generate_cross_attnw_layer(
    module, inputs, key_padding=None, attn_mask=None
):
    x, y = module.multihead_attn(
        *inputs, attn_mask=attn_mask,
        key_padding_mask=key_padding
    )
    return y


if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--token_path', required=True)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    fidx = 0

    while True:
        f_name = f'sample_{fidx}.pkl'
        if not os.path.exists(os.path.join(args.output_dir, f_name)):
            break
        fidx += 1

    out_dir = os.path.join(args.output_dir, f'sample_{fidx}.pkl')

    if torch.cuda.is_available() and args.device > 0:
        device = torch.device(f'cuda:{args.device}')
    else:
        device = torch.device('cpu')

    with open(args.token_path, 'rb') as Fin:
        tokenizer = pickle.load(Fin)

    DIM, LAY, HEAD = 512, 8, 8
    neg_slope, n_cls = 0.2, None

    GNN = GATBase(
        num_layers=LAY, dropout=0.1, embedding_dim=DIM,
        num_heads=HEAD, negative_slope=neg_slope, n_class=n_cls
    )

    decode_layer = TransformerDecoderLayer(
        d_model=DIM, nhead=HEAD, batch_first=True,
        dim_feedforward=DIM * 2, dropout=0.1
    )
    Decoder = TransformerDecoder(decode_layer, LAY)
    Pos_env = PositionalEncoding(DIM, 0.1, maxlen=2000)

    model = PretrainModel(
        token_size=tokenizer.get_token_size(), encoder=GNN,
        decoder=Decoder, d_model=DIM, pos_enc=Pos_env
    )

    weight = torch.load(args.model_path, map_location=device)

    model.load_state_dict(weight, strict=False)

    model = model.eval().to(device)

    crs_ips = []
    crs_ops = []

    for i in range(LAY):
        model.decoder.layers[i].multihead_attn.register_forward_hook(
            lambda x, y, z: fwd_hood(x, y, z, crs_ips, crs_ops)
        )
    # rxn = '[CH3:2][C:19]([CH3:3])([CH3:4])[O:12][C:14](=[O:6])[O:24][C:26](=[O:23])[O:25][C:27]([CH3:20])([CH3:21])[CH3:22].[CH3:1][C:13](=[O:5])[c:15]1[cH:7][cH:10][c:17]2[c:16]([cH:8][cH:11][nH:18]2)[cH:9]1>>[CH3:1][C:13](=[O:5])[c:15]1[cH:7][cH:10][c:17]2[c:16]([cH:8][cH:11][n:18]2[C:14](=[O:6])[O:12][C:19]([CH3:2])([CH3:3])[CH3:4])[cH:9]1'
    # rxn = '[CH2:4]1[CH2:5][CH:12]1[CH2:6][NH2:9].[Cl:1][c:10]1[cH:2][n:7][cH:3][c:11]([Cl:13])[n:8]1>>[Cl:1][c:10]1[cH:2][n:7][cH:3][c:11]([NH:9][CH2:6][CH:12]2[CH2:4][CH2:5]2)[n:8]1'
    # rxn = '[CH3:2][O:19][C:22](=[O:7])[C@@H:27]1[CH2:14][N:28]([C:23](=[O:8])[O:20][C:30]([CH3:3])([CH3:4])[CH3:5])[CH2:13][CH2:15][NH:29]1.[CH3:1][O:18][C:21](=[O:6])[c:25]1[cH:12][n:16][c:24]([Cl:32])[n:17][c:26]1[C:31]([F:9])([F:10])[F:11]>>[CH3:1][O:18][C:21](=[O:6])[c:25]1[cH:12][n:16][c:24]([N:29]2[CH2:15][CH2:13][N:28]([C:23](=[O:8])[O:20][C:30]([CH3:3])([CH3:4])[CH3:5])[CH2:14][C@H:27]2[C:22](=[O:7])[O:19][CH3:2])[n:17][c:26]1[C:31]([F:9])([F:10])[F:11]'
    # rxn = '[CH3:1][O:9][c:12]1[c:11]([CH3:2])[cH:5][cH:6][c:13]2[nH:8][c:10]([C:4]([NH2:3])=[O:15])[cH:7][c:14]12>>[CH3:1][O:9][c:12]1[c:11]([CH3:2])[cH:5][cH:6][c:13]2[nH:8][c:10]([C:4]#[N:3])[cH:7][c:14]12'
    # rxn = '[F:1][c:13]1[cH:4][cH:7][c:15]([CH2:11][Br:17])[cH:8][cH:5]1.[Br:2][c:14]1[cH:6][cH:3][cH:9][c:16]([SH:12])[cH:10]1>>[F:1][c:13]1[cH:4][cH:7][c:15]([CH2:11][S:12][c:16]2[cH:9][cH:3][cH:6][c:14]([Br:2])[cH:10]2)[cH:8][cH:5]1'
    # rxn = '[CH3:1][C:11](=[O:2])[O:16][C:17]([CH3:14])=[O:15].[S:3]=[C:12]([CH2:7][CH2:8][c:13]1[cH:6][cH:4][cH:5][o:10]1)[NH2:9]>>[CH3:1][C:11](=[O:2])[NH:9][C:12](=[S:3])[CH2:7][CH2:8][c:13]1[cH:6][cH:4][cH:5][o:10]1'
    rxn = '[S:2]=[C:5]([Cl:13])[Cl:14].[F:1][c:10]1[cH:6][c:9]([NH2:8])[cH:7][c:11]([Cl:3])[c:12]1[Br:4]>>[F:1][c:10]1[cH:6][c:9]([N:8]=[C:5]=[S:2])[cH:7][c:11]([Cl:3])[c:12]1[Br:4]'

    reac, prod = rxn.split('>>')

    x_prod = cano_with_am(prod)

    reac = process_reac_via_prod(x_prod, reac)
    graph = make_graph_batch(x_prod).to(device)

    reac_tokens = ['<CLS>'] + smi_tokenizer(remove_am_wo_cano(reac))

    token_ip = tokenizer.encode2d([reac_tokens])
    token_ip = torch.LongTensor(token_ip).to(device)

    pad_mask, sub_mask = generate_tgt_mask(
        token_ip, tokenizer, '<PAD>', device
    )

    model(
        graphs=graph, tgt=token_ip, tgt_mask=sub_mask,
        tgt_pad_mask=pad_mask,
    )

    crs_map = []
    for i in range(LAY):
        crs = generate_cross_attnw_layer(
            model.decoder.layers[i], crs_ips[i]
        )
        crs_map.append(crs.tolist()[0])

    with open(out_dir, 'wb') as Fout:
        pickle.dump({
            'query': rxn, 'crs_map': crs_map,
            'prod': x_prod, 'reac': reac,
            'reac_tokens': reac_tokens,
            'prod_order': find_all_amap(x_prod)
        }, Fout)
