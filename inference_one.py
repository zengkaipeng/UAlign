from tqdm import tqdm
import torch
import argparse
import json
import pickle


from torch.utils.data import DataLoader
from model import PretrainModel, PositionalEncoding
from data_utils import fix_seed
from torch.nn import TransformerDecoderLayer, TransformerDecoder
from sparse_backBone import GATBase
from utils.chemistry_parse import clear_map_number
from utils.graph_utils import smiles2graph
import pandas
import torch_geometric
from inference_tools import beam_search_one
import time
import os


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Inference part')
    parser.add_argument(
        '--dim', default=256, type=int,
        help='the hidden dim of model'
    )
    parser.add_argument(
        '--n_layer', default=8, type=int,
        help='the layer of encoder gnn'
    )
    parser.add_argument(
        '--heads', default=4, type=int,
        help='the number of heads for attention, only useful for gat'
    )
    parser.add_argument(
        '--negative_slope', type=float, default=0.2,
        help='negative slope for attention, only useful for gat'
    )
    parser.add_argument(
        '--data_path', required=True, type=str,
        help='the path containing dataset'
    )
    parser.add_argument(
        '--seed', type=int, default=2023,
        help='the seed for training'
    )

    parser.add_argument(
        '--device', default=-1, type=int,
        help='the device for running exps'
    )
    parser.add_argument(
        '--checkpoint', type=str, required=True,
        help='the path of checkpoint to restart the exp'
    )
    parser.add_argument(
        '--token_ckpt', type=str, required=True,
        help='the path of tokenizer, when ckpt is loaded, necessary'
    )
    parser.add_argument(
        '--use_class', action='store_true',
        help='use the class for model or not'
    )
    parser.add_argument(
        '--max_len', default=300, type=int,
        help='the max num of tokens in result'
    )
    parser.add_argument(
        '--beams', default=10, type=int,
        help='the number of beams '
    )
    parser.add_argument(
        '--output_folder', default='results', type=str,
        help='the path containing results'
    )
    parser.add_argument(
        '--save_every', type=int, default=1000,
        help='the step to save result into file'
    )

    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--len', type=int, default=-1)

    args = parser.parse_args()
    print(args)

    if not torch.cuda.is_available() or args.device < 0:
        device = torch.device('cpu')
    else:
        device = torch.device(f'cuda:{args.device}')

    fix_seed(args.seed)
    with open(args.token_ckpt, 'rb') as Fin:
        tokenizer = pickle.load(Fin)

    GNN = GATBase(
        num_layers=args.n_layer, dropout=0.1, embedding_dim=args.dim,
        num_heads=args.heads, negative_slope=args.negative_slope,
        n_class=11 if args.use_class else None
    )

    decode_layer = TransformerDecoderLayer(
        d_model=args.dim, nhead=args.heads, batch_first=True,
        dim_feedforward=args.dim * 2, dropout=0.1
    )
    Decoder = TransformerDecoder(decode_layer, args.n_layer)
    Pos_env = PositionalEncoding(args.dim, 0.1, maxlen=2000)

    model = PretrainModel(
        token_size=tokenizer.get_token_size(), encoder=GNN,
        decoder=Decoder, d_model=args.dim, pos_enc=Pos_env
    ).to(device)

    if args.checkpoint != '':
        assert args.token_ckpt != '', 'Missing Tokenizer Information'
        print(f'[INFO] Loading model weight in {args.checkpoint}')
        weight = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(weight, strict=True)

    print('[INFO] padding index', tokenizer.token2idx['<PAD>'])

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    meta_df = pandas.read_csv(args.data_path)

    if args.len > 0:
        end_pos = min(len(meta_df), args.start + args.len)
        file_name = f'{args.start}-{end_pos}.json'
        out_file = os.path.join(args.output_folder, file_name)
        meta_df = meta_df[args.start: end_pos]
    else:
        file_name = f'{args.start}-{len(meta_df)}.json'
        out_file = os.path.join(args.output_folder, file_name)

    answers = []

    for idx, resu in enumerate(tqdm(meta_df['reactants>reagents>production'])):
        rea, prd = resu.strip().split('>>')
        prd = clear_map_number(prd)
        rea = clear_map_number(rea)
        if args.use_class:
            rxn_class = int(meta_df['class'][idx])
            start_token = f'<RXN>_{rxn_class}'
        else:
            rxn_class = None
            start_token = '<CLS>'

        g_ip = make_graph_batch(prd, rxn_class).to(device)

        preds, probs = beam_search_one(
            model, tokenizer, g_ip, device, max_len=args.max_len,
            size=args.beams, begin_token=start_token, end_token='<END>',
            pen_para=0, validate=False
        )

        answers.append({
            'query': resu, 'idx': idx, 'rxn_class': rxn_class,
            'answer': preds, 'prob': probs
        })
        if idx % args.save_every == 0:
            with open(out_file, 'w') as Fout:
                json.dump({
                    'args': args.__dict__,
                    'answer': answers
                }, Fout, indent=4)

    with open(out_file, 'w') as Fout:
        json.dump({
            'args': args.__dict__,
            'answer': answers
        }, Fout, indent=4)
