import torch
import argparse
import json
import pickle
import numpy as np


from models.ualign import PretrainModel, PositionalEncoding
from utils.data_utils import fix_seed
from models.decoder import CachedTransformerDecoder, CachedTransformerDecoderLayer
from models.sparse_backBone import GATBase
from utils.chemistry_parse import canonical_smiles
from utils.graph_utils import smiles2graph
import torch_geometric
from rdkit import Chem
from utils.inference_tools import beam_search_batch, merge_prediction_group


def get_augmented_products(smi, aug_time):
    prod = canonical_smiles(smi)
    prod_mol = Chem.MolFromSmiles(prod)
    products = [Chem.MolToSmiles(prod_mol)]
    for _ in range(aug_time - 1):
        products.append(Chem.MolToSmiles(prod_mol, doRandom=True))
    return products


def make_graph_batch(products, rxn=None):
    if isinstance(products, str):
        products = [products]

    batch_size, max_node = len(products), 0
    edge_idxes, edge_feats, node_feats, lstnode = [], [], [], 0
    batch, ptr, node_per_graph = [], [0], []
    node_rxn, edge_rxn = [], []

    for idx, smi in enumerate(products):
        graph = smiles2graph(smi, with_amap=False)
        num_nodes = graph['node_feat'].shape[0]
        num_edges = graph['edge_index'].shape[1]

        edge_idxes.append(graph['edge_index'] + lstnode)
        edge_feats.append(graph['edge_feat'])
        node_feats.append(graph['node_feat'])

        lstnode += num_nodes
        max_node = max(max_node, num_nodes)
        node_per_graph.append(num_nodes)
        batch.append(torch.ones(num_nodes, dtype=torch.long) * idx)
        ptr.append(lstnode)

        if rxn is not None:
            node_rxn.append(torch.ones(num_nodes, dtype=torch.long) * rxn)
            edge_rxn.append(torch.ones(num_edges, dtype=torch.long) * rxn)

    data = {
        'x': torch.from_numpy(np.concatenate(node_feats, axis=0)),
        'num_nodes': lstnode,
        'edge_attr': torch.from_numpy(np.concatenate(edge_feats, axis=0)),
        'edge_index': torch.from_numpy(np.concatenate(edge_idxes, axis=-1)),
        'ptr': torch.LongTensor(ptr),
        'batch': torch.cat(batch, dim=0),
    }
    all_batch_mask = torch.zeros((batch_size, max_node))
    for idx, mk in enumerate(node_per_graph):
        all_batch_mask[idx, :mk] = 1
    data['batch_mask'] = all_batch_mask.bool()

    if rxn is not None:
        data['node_rxn'] = torch.cat(node_rxn, dim=0)
        data['edge_rxn'] = torch.cat(edge_rxn, dim=0)

    return torch_geometric.data.Data(**data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Graph Edit Exp, Sparse Model')
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
        '--product_smiles', type=str, required=True,
        help='the SMILES of product, containing only one mole'
    )
    parser.add_argument(
        '--input_class', type=int, default=-1,
        help='the input class for reaction, required when' +
        ' use_class option is chosen'
    )
    parser.add_argument(
        '--org_output', action='store_true',
        help='preserve the original output,' +
        ' if chosen the invalid smiles will not be removed'
    )
    parser.add_argument(
        '--aug_time', type=int, default=1,
        help='the number of product SMILES augmentations for test-time inference'
    )

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

    decode_layer = CachedTransformerDecoderLayer(
        d_model=args.dim, nhead=args.heads, batch_first=True,
        dim_feedforward=args.dim * 2, dropout=0.1
    )
    Decoder = CachedTransformerDecoder(decode_layer, args.n_layer)
    Pos_env = PositionalEncoding(args.dim, 0.1, maxlen=2000)

    model = PretrainModel(
        token_size=tokenizer.get_token_size(), encoder=GNN,
        decoder=Decoder, d_model=args.dim, pos_enc=Pos_env
    ).to(device)

    if args.checkpoint != '':
        assert args.token_ckpt != '', 'Missing Tokenizer Information'
        print(f'[INFO] Loading model weight in {args.checkpoint}')
        weight = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(weight, strict=False)

    print('[INFO] padding index', tokenizer.token2idx['<PAD>'])
    if args.use_class:
        assert args.input_class != -1, 'require reaction class!'
        start_token, rxn_class = f'<RXN>_{args.input_class}', args.input_class
    else:
        start_token, rxn_class = '<CLS>', None

    products = get_augmented_products(args.product_smiles, args.aug_time)
    g_ip = make_graph_batch(products, rxn_class).to(device)

    pred_batch, prob_batch = beam_search_batch(
        model=model,
        tokenizer=tokenizer,
        graphs=g_ip,
        device=device,
        begin_tokens=[start_token] * len(products),
        max_len=args.max_len,
        size=args.beams,
        pen_para=0,
        validate=not args.org_output,
    )
    preds, probs = merge_prediction_group(
        pred_batch, prob_batch, keep_invalid=args.org_output
    )

    print('[RESULT]')
    print(json.dumps({
        "answers": preds, 'probs': probs,
        'rxn_class': args.input_class,
        'aug_time': args.aug_time,
    }, indent=4))


