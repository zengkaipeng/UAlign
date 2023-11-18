import torch_geometric
from ogb.utils import smiles2graph
from torch.nn import TransformerDecoderLayer, TransformerDecoder
import pickle
from inference_tools import greedy_inference_one
from model import Graph2Seq, PositionalEncoding
from MixConv import MixFormer
import torch

MODEL_PATH, TOKEN_PATH = 'MODEL.pth', 'TOKEN.pkl'
HEADS, ENCODER_LAYER, DECODER_LAYER = 6, 8, 8
DIM, DEVICE = 768, 0


def smiles_to_graph_input(smiles):
    graph = smiles2graph(smiles)
    n_node = graph['num_nodes']
    data = {
        'ptr': torch.LongTensor([0, n_node]),
        'batch': torch.zeros(n_node).long(),
        'edge_index': torch.from_numpy(graph['edge_index']),
        'edge_attr': torch.from_numpy(graph['edge_feat']),
        'x': torch.from_numpy(graph['node_feat']),
        'num_nodes': n_node,
        'batch_mask': torch.ones(1, n_node).bool()
    }
    return torch_geometric.data.Data(**data)


if not torch.cuda.is_available() or DEVICE < 0:
    device = torch.device('cpu')
else:
    device = torch.device(f'cuda:{DEVICE}')


with open(TOKEN_PATH, 'rb') as Fin:
    tokenizer = pickle.load(Fin)

GNN = MixFormer(
    emb_dim=DIM, num_layer=ENCODER_LAYER, heads=HEADS,
    dropout=0.1, negative_slope=0.2,
)

decode_layer = TransformerDecoderLayer(
    d_model=DIM, nhead=HEADS * 2, batch_first=True,
    dim_feedforward=DIM * 2, dropout=0.1
)
Decoder = TransformerDecoder(decode_layer, DECODER_LAYER)
Pos_env = PositionalEncoding(DIM, 0.1, maxlen=2000)


model = Graph2Seq(
    token_size=tokenizer.get_token_size(), encoder=GNN,
    decoder=Decoder, d_model=DIM, pos_enc=Pos_env
).to(device)

state_dict = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state_dict)
model = model.eval()


example_smile = 'C[Si](C)(C)OC(=O)/C=C/CBr'
graph_input = smiles_to_graph_input(example_smile).to(device)

print(beam_search_one(
    model, tokenizer, graph_input, device, 300, size=10,
    begin_token='<CLS>', validate=True, end_token='<END>'
))
