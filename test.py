import torch
from tokenlizer import DEFAULT_SP, Tokenizer
from torch.utils.data import DataLoader
from sparse_backBone import GINBase, GATBase
from model import Graph2Seq, fc_collect_fn, PositionalEncoding
from inference_tools import greedy_inference_one
import pickle
from data_utils import create_sparse_dataset, load_data, fix_seed
from torch.nn import TransformerDecoderLayer, TransformerDecoder
from tqdm import tqdm

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
    test_set = create_sparse_dataset(
        test_rec, test_prod, kekulize=args.kekulize,
        rxn_class=test_rxn if args.use_class else None
    )
    test_loader = DataLoader(
        test_set, collate_fn=fc_collect_fn,
        batch_size=1, shuffle=False
    )

    if args.backbone == 'GIN':
        GNN = GINBase(
            num_layers=args.layer_encoder, dropout=args.dropout,
            embedding_dim=args.dim, edge_last=True, residual=True
        )
    else:
        GNN = GATBase(
            num_layers=args.layer_encoder, dropout=args.dropout,
            embedding_dim=args.dim, edge_last=True,
            residual=True, negative_slope=args.negative_slope,
            num_heads=args.heads, add_self_loop=True
        )

    decode_layer = TransformerDecoderLayer(
        d_model=args.dim, nhead=args.heads, batch_first=True,
        dim_feedforward=args.dim * 2, dropout=args.dropout
    )
    Decoder = TransformerDecoder(decode_layer, args.layer_decoder)
    Pos_env = PositionalEncoding(args.dim, args.dropout, maxlen=2000)

    model = Graph2Seq(
        token_size=tokenizer.get_token_size(), encoder=GNN,
        decoder=Decoder, d_model=args.dim, pos_enc=Pos_env
    ).to(device)

    model = model.eval()

    state_dict = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state_dict)

    for data in tqdm(test_loader):
        graphs, gt = data
        graphs = graphs.to(device)
        if hasattr(graphs, 'rxn_class'):
            rxn_class = graphs.rxn_class.item()
        else:
            rxn_class = None

        result = greedy_inference_one(
            model, tokenizer, graphs, device, args.max_len,
            begin_token=f'<RXN_{rxn_class}>' if args.use_class else '<CLS>'
        )
        print(result, gt)
        exit()
