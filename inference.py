from model import OverallModel
from data_utils import avg_edge_logs
import argparse
import pandas

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Parser')
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

    args = parser.parse_args()

    meta_file = pandas.read_csv(args.file)

    