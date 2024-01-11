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
    args = parser.parse_args()

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

    ft_ips = []
    ft_ops = []

    for i in range(LAY):
        model.decoder.layers[i].multihead_attn.register_forward_hook(
            lambda x, y, z: fwd_hood(x, y, z, ft_ips, ft_ops)
        )

    rxn = '[CH:2](=[O:3])[N:10]([CH3:11])[CH3:12].[c:1]1([Br:13])[cH:4][cH:5][c:6]([Br:7])[n:8][cH:9]1>>[c:1]1([CH:2]=[O:3])[cH:4][cH:5][c:6]([Br:7])[n:8][cH:9]1'

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
