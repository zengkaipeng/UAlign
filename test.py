import torch
from tokenlizer import DEFAULT_SP, Tokenizer
from torch.utils.data import DataLoader
from sparse_backBone import GINBase, GATBase
from model import Graph2Seq, fc_collect_fn, PositionalEncoding
import pickle
from data_utils import create_sparse_dataset, load_data, fix_seed
from torch.nn import TransformerDecoderLayer, TransformerDecoder

if __name__ == '__main__':
	parser = argparse.ArgumentParser('Graph Edit Exp, Sparse Model')
    parser.add_argument(
        '--dim', default=256, type=int,
        help='the hidden dim of model'
    )
    parser.add_argument(
        '--kekulize', action='store_true',
        help='kekulize molecules if it\'s added'
    )
    parser.add_argument(
        '--layer_encoder', default=8, type=int,
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
        '--backbone', type=str, choices=['GAT', 'GIN'],
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

    args = parser.parse_args()
    print(args)

    with open(args.token_path, 'rb') as Fin:
    	tokenizer = pickle.load(Fin)
    if not torch.cuda.is_available() or args.device < 0:
        device = torch.device('cpu')
    else:
        device = torch.device(f'cuda:{args.device}')
    fix_seed(args.seed)
    test_rec, test_prod, test_rxn = load_data(args.data_path)

