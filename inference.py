from model import OverallModel
from sparse_backBone import GINBase, GATBase
from Mix_backbone import MixFormer
import argparse
import pandas
import pickle
import torch
from inference_tools import beam_seach_one
from utils.chemistry_parse import clear_map_number
import numpy as np
from data_utils import fix_seed
from tqdm import tqdm


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Parser for beam serach inference')
    parser.add_argument(
        '--dim', default=256, type=int,
        help='the hidden dim of model'
    )
    parser.add_argument(
        '--n_layer', default=5, type=int,
        help='the layer of backbones'
    )
    parser.add_argument(
        '--file', required=True, type=str,
        help='the path containing dataset'
    )
    parser.add_argument(
        '--use_class', action='store_true',
        help='use rxn_class for training or not'
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
        '--gnn_type', type=str, choices=['gat', 'gin'],
        help='type of gnn backbone', required=True
    )

    # GAT & Gtrans setting
    parser.add_argument(
        '--transformer', action='store_true',
        help='use graph transformer or not'
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
        '--update_gate', choices=['cat', 'add'], default='add',
        help='the update method for mixformer', type=str,
    )
    parser.add_argument(
        '--checkpoint', required=True, type=str,
        help='the path of trained overall model, to restart the exp'
    )
    parser.add_argument(
        '--pregraph', action='store_true',
        help='cat the graph embedding before the transformer ' +
        'encoder, if not chosen, cat it after transformer encoder'
    )
    parser.add_argument(
        '--use_sim', action='store_true',
        help='use the sim model while making link prediction ' +
        'between leaving groups and synthons'
    )
    parser.add_argument(
        '--token_ckpt', type=str, required=True,
        help='the path of token checkpoint, required while' +
        ' checkpoint is specified'
    )
    parser.add_argument(
        '--beam', type=int, default=10,
        help='the beam size for beam search'
    )

    parser.add_argument(
        '--max_len', type=int, default=100,
        help='the max_length for generating the leaving group'
    )

    args = parser.parse_args()

    meta_file = pandas.read_csv(args.file)

    if not torch.cuda.is_available() or args.device < 0:
        device = torch.device('cpu')
    else:
        device = torch.device(f'cuda:{args.device}')

    fix_seed(args.seed)
    with open(args.token_ckpt, 'rb') as Fin:
        tokenizer = pickle.load(Fin)

    if args.transformer:
        if args.gnn_type == 'gin':
            gnn_args = {
                'in_channels': args.dim, 'out_channels': args.dim,
                'edge_dim': args.dim
            }
        elif args.gnn_type == 'gat':
            assert args.dim % args.heads == 0, \
                'The model dim should be evenly divided by num_heads'
            gnn_args = {
                'in_channels': args.dim, 'dropout': 0.1,
                'out_channels': args.dim // args.heads, 'edge_dim': args.dim,
                'negative_slope': args.negative_slope, 'heads': args.heads
            }
        else:
            raise ValueError(f'Invalid GNN type {args.backbone}')

        GNN = MixFormer(
            emb_dim=args.dim, n_layers=args.n_layer, gnn_args=gnn_args,
            dropout=0.1, heads=args.heads, gnn_type=args.gnn_type,
            n_class=11 if args.use_class else None,
            update_gate=args.update_gate
        )
    else:
        if args.gnn_type == 'gin':
            GNN = GINBase(
                num_layers=args.n_layer, dropout=0.1,
                embedding_dim=args.dim, n_class=11 if args.use_class else None
            )
        elif args.gnn_type == 'gat':
            GNN = GATBase(
                num_layers=args.n_layer, dropout=0.1,
                embedding_dim=args.dim, num_heads=args.heads,
                negative_slope=args.negative_slope,
                n_class=11 if args.use_class else None
            )
        else:
            raise ValueError(f'Invalid GNN type {args.backbone}')

    enc_layer = torch.nn.TransformerEncoderLayer(
        args.dim, args.heads, dim_feedforward=args.dim * 2,
        batch_first=True, dropout=0.1
    )
    dec_layer = torch.nn.TransformerDecoderLayer(
        args.dim, args.heads, dim_feedforward=args.dim * 2,
        batch_first=True, dropout=0.1
    )
    TransEnc = torch.nn.TransformerEncoder(enc_layer, args.n_layer)
    TransDec = torch.nn.TransformerDecoder(dec_layer, args.n_layer)

    model = OverallModel(
        GNN, TransEnc, TransDec, args.dim, args.dim,
        num_token=tokenizer.get_token_size(), heads=args.heads,
        dropout=0.1, use_sim=args.use_sim, pre_graph=args.pregraph,
        rxn_num=11 if args.use_class else None
    ).to(device)

    model_weight = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(model_weight)
    model = model.eval()

    topks = []

    for idx, resu in enumerate(tqdm(meta_file['reactants>reagents>production'])):
        if args.use_class:
            rxn_class = meta_file['class'][idx]
            start_toekn = f'<RXN_{rxn_class}>'
        else:
            rxn_class = None
            start_toekn = "<CLS>"
        reac, prod = resu.strip().split('>>')

        answer = clear_map_number(reac)

        preds = beam_seach_one(
            smiles=prod, model=model, tokenizer=tokenizer, device=device,
            beam_size=args.beam, rxn=rxn_class, start_token=start_toekn,
            end_token='<END>', sep_token='`', max_len=args.max_len
        )

        this_hit = np.zeros(args.beam)

        for idx, (res, score) in enumerate(preds):
            if res == answer:
                this_hit[idx:] = 1
                break

        topks.append(this_hit)

    topks = np.stack(topks, axis=0)

    topk_acc = np.mean(topks, axis=0)
    for i in [1, 3, 5, 10]:
        print(f'[TOP {i}]', topk_acc[i - 1])
