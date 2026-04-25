import torch
import argparse
import json
import pickle
import numpy as np


from models import PretrainModel, load_model_arch
from utils.data_utils import fix_seed
from utils.chemistry_parse import augment_product_smiles, clear_map_number
from utils.graph_utils import smiles2graph
import torch_geometric
from rdkit import Chem
from utils.inference_tools import beam_search_batch
from utils.rerank import rerank_predictions


def get_augmented_products(smi, aug_time):
    prod = clear_map_number(smi)
    products = [augment_product_smiles(prod, do_random=False)]
    for _ in range(aug_time - 1):
        products.append(augment_product_smiles(prod, do_random=True))
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
        '--model_arch_path', required=True, type=str,
        help='the path of model architecture json'
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
        help='the SMILES of product'
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
    parser.add_argument(
        '--disable_kv_cache', action='store_true',
        help='disable KV cache and recompute the decoder state each step'
    )
    parser.add_argument(
        '--rank_compute', type=str, default='log_logits',
        choices=['log_logits', 'ensemble'],
        help=(
            'how to combine augmented predictions: '
            'log_logits sums probabilities, ensemble uses reciprocal-rank '
            'voting after per-augmentation canonical merging'
        )
    )
    parser.add_argument(
        '--score_alpha', type=float, default=0.1,
        help='alpha used by ensemble reciprocal-rank scoring'
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
    args.model_arch = load_model_arch(args.model_arch_path)

    model = PretrainModel.from_arch(
        token_size=tokenizer.get_token_size(),
        model_arch=args.model_arch,
        use_class=args.use_class,
    ).to(device)

    if args.checkpoint != '':
        assert args.token_ckpt != '', 'Missing Tokenizer Information'
        print(f'[INFO] Loading model weight in {args.checkpoint}')
        weight = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(weight, strict=False)
    model.eval()

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
        use_kv_cache=not args.disable_kv_cache,
    )
    preds, probs = rerank_predictions(
        pred_batch,
        prob_batch,
        rank_compute=args.rank_compute,
        keep_invalid=args.org_output,
        score_alpha=args.score_alpha,
    )

    print('[RESULT]')
    print(json.dumps({
        "answers": preds, 'probs': probs,
        'rxn_class': args.input_class,
        'aug_time': args.aug_time,
    }, indent=4))


