import torch
from tokenlizer import DEFAULT_SP, Tokenizer
from torch.utils.data import DataLoader
from sparse_backBone import GINBase, GATBase
from model import Graph2Seq, col_fn_unloop, PositionalEncoding
from model import OnFlyDataset
from inference_tools import beam_search_one, check_valid
import pickle
from data_utils import load_ext_data, fix_seed
from torch.nn import TransformerDecoderLayer, TransformerDecoder
from tqdm import tqdm
from utils.chemistry_parse import canonical_smiles
import argparse
import numpy as np
from MixConv import MixFormer

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Graph Edit Exp, Sparse Model')
    parser.add_argument(
        '--dim', default=288, type=int,
        help='the hidden dim of model'
    )
    parser.add_argument(
        '--layer_encoder', default=10, type=int,
        help='the layer of encoder gnn'
    )
    parser.add_argument(
        '--layer_decoder', default=8, type=int,
        help='the layer of transformer decoder'
    )
    parser.add_argument(
        '--token_path', required=True, type=str,
        help='the path of a json containing all tokens'
    )
    parser.add_argument(
        '--heads', default=4, type=int,
        help='the number of heads for attention, only useful for gat'
    )
    parser.add_argument(
        '--backbone', type=str, choices=['GAT', 'GIN', 'MIX'],
        help='type of gnn backbone', required=True
    )
    parser.add_argument(
        '--dropout', type=float, default=0.3,
        help='the dropout rate, useful for all backbone'
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
        '--model_path', required=True, type=str,
        help='the path containing the pretrained model'
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
        '--max_len', default=200, type=int,
        help='the max length for decoding'
    )
    parser.add_argument(
        '--kekulize', action='store_true', 
        help='kekulize the mole'
    )

    args = parser.parse_args()
    print(args)

    with open(args.token_path, 'rb') as Fin:
        tokenizer = pickle.load(Fin)
    if not torch.cuda.is_available() or args.device < 0:
        device = torch.device('cpu')
    else:
        device = torch.device(f'cuda:{args.device}')
    fix_seed(args.seed)
    test_rec, test_prod, test_rxn, test_target =\
        load_ext_data(args.data_path)
    test_set = OnFlyDataset(
        prod_sm=test_prod, reat_sm=test_rec, target=test_target,
        aug_prob=0, randomize=False, kekulize=args.kekulize
    )
    test_loader = DataLoader(
        test_set, collate_fn=col_fn_unloop,
        batch_size=1, shuffle=False
    )

    if args.backbone == 'GIN':
        GNN = GINBase(
            num_layers=args.layer_encoder, dropout=args.dropout,
            embedding_dim=args.dim,
        )
    elif args.backbone == 'GAT':
        GNN = GATBase(
            num_layers=args.layer_encoder, dropout=args.dropout,
            embedding_dim=args.dim, negative_slope=args.negative_slope,
            num_heads=args.heads,
        )
    else:
        GNN = MixFormer(
            emb_dim=args.dim, num_layer=args.layer_encoder,
            heads=args.heads, dropout=args.dropout,
            negative_slope=args.negative_slope,
        )

    trans_head = args.heads if args.backbone != 'MIX' else args.heads * 2
    decode_layer = TransformerDecoderLayer(
        d_model=args.dim, nhead=trans_head, batch_first=True,
        dim_feedforward=args.dim * 2, dropout=args.dropout
    )
    Decoder = TransformerDecoder(decode_layer, args.layer_decoder)
    Pos_env = PositionalEncoding(args.dim, args.dropout, maxlen=2000)

    model = Graph2Seq(
        token_size=tokenizer.get_token_size(), encoder=GNN,
        decoder=Decoder, d_model=args.dim, pos_enc=Pos_env
    ).to(device)

    state_dict = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.eval()

    topk_acc = []
    for data in tqdm(test_loader):
        graphs, gt = data
        graphs = graphs.to(device)
        if hasattr(graphs, 'rxn_class'):
            rxn_class = graphs.rxn_class.item()
        else:
            rxn_class = None

        result, prob = beam_search_one(
            model, tokenizer, graphs, device, args.max_len, size=10,
            begin_token='<CLS>', validate=False, end_token='<END>'
        )

        accs, gts = np.zeros(10), ''.join(gt[0][1: -1])

        for idx, t in enumerate(result):
            if check_valid(t):
                t = canonical_smiles(t)
            if t == gts:
                accs[idx:] = 1
                break
        topk_acc.append(accs)

    topk_acc_result = np.mean(topk_acc, axis=0)
    for tdx in [1, 3, 5, 10]:
        print(f'[TOP{tdx}-ACC]', topk_acc_result[tdx - 1])
