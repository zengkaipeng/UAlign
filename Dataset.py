import torch

from tokenlizer import smi_tokenizer
from utils.graph_utils import smiles2graph


class TransDataset(torch.utils.data.Dataset):
    def __init__(self, smiles, reacts, mode='train'):
        super(TransDataset, self).__init__()
        self.smiles = smiles
        self.reacts = reacts
        self.mode = mode
        self.offset = len(self.smiles)

        assert mode in ['train', 'eval'], f'Invalid mode {mode}'

    def __len__(self):
        return len(self.smiles) + len(self.reacts)

    def randomize_smiles(self, smi):
        if random.randint(0, 1) == 1:
            k = random.choice(self.smiles)
            return f'{smi}.{k}'
        else:
            mol = Chem.MolFromSmiles(smi)
            return Chem.MolToSmiles(mol, doRandom=True)

    def random_react(self, smi):
        y = []
        for x in smi.split('.'):
            mol = Chem.MolFromSmiles(smi)
            y.append(Chem.MolToSmiles(mol, doRandom=True))
        return '.'.join(y)

    def __getitem__(self, index):
        ret = ['<CLS>']
        out_smi = self.smiles[index] if index < self.offset \
            else self.reacts[index - self.offset]
        if self.mode == 'train':
            out_smi = self.randomize_smiles(out_smi)

        ret.extend(smi_tokenizer(out_smi))
        ret.append('<END>')

        return smiles2graph(out_smi, with_amap=False), ret


def col_fn_pretrain(data_batch):
    batch_size, max_node = len(data_batch), 0
    edge_idxes, edge_feats, node_feats, lstnode = [], [], [], 0
    batch, ptr, reats, node_per_graph = [], [0], [], []
    for idx, data in enumerate(data_batch):
        graph, ret = data
        num_nodes = graph['num_nodes']
        num_edges = graph['edge_index'].shape[1]
        reats.append(ret)

        edge_idxes.append(graph['edge_index'] + lstnode)
        edge_feats.append(graph['edge_feat'])
        node_feats.append(graph['node_feat'])

        lstnode += num_nodes
        max_node = max(max_node, num_nodes)
        node_per_graph.append(num_nodes)
        batch.append(np.ones(num_nodes, dtype=np.int64) * idx)
        ptr.append(lstnode)

    result = {
        'edge_index': np.concatenate(edge_idxes, axis=-1),
        'edge_attr': np.concatenate(edge_feats, axis=0),
        'batch': np.concatenate(batch, axis=0),
        'x': np.concatenate(node_feats, axis=0),
        'ptr': np.array(ptr, dtype=np.int64)
    }

    result = {k: torch.from_numpy(v) for k, v in result.items()}
    result['num_nodes'] = lstnode

    all_batch_mask = torch.zeros((batch_size, max_node))
    for idx, mk in enumerate(node_per_graph):
        all_batch_mask[idx, :mk] = 1
    result['batch_mask'] = all_batch_mask.bool()

    return GData(**result), reats
