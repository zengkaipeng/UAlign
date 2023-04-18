from ogb.utils.features import (
    allowable_features, atom_to_feature_vector, bond_feature_vector_to_dict,
    bond_to_feature_vector, atom_feature_vector_to_dict
)
import torch


def graph_batch_from_smiles(smiles_list, device=torch.device('cpu')):
    edge_idxes, edge_feats, node_feats, lstnode, batch = [], [], [], 0, []
    graphs = [smiles2graph(x) for x in smiles_list]
    for idx, graph in enumerate(graphs):
        edge_idxes.append(graph['edge_index'] + lstnode)
        edge_feats.append(graph['edge_feat'])
        node_feats.append(graph['node_feat'])
        lstnode += graph['num_nodes']
        batch.append(np.ones(graph['num_nodes'], dtype=np.int64) * idx)

    result = {
        'edge_index': np.concatenate(edge_idxes, axis=-1),
        'edge_attr': np.concatenate(edge_feats, axis=0),
        'batch': np.concatenate(batch, axis=0),
        'x': np.concatenate(node_feats, axis=0)
    }
    result = {k: torch.from_numpy(v).to(device) for k, v in result.items()}
    result['num_nodes'] = lstnode
    result['num_edges'] = result['edge_index'].shape[1]
    return result


def smiles2graph(smiles_string, with_amap=False):
    """
    Converts SMILES string to graph Data object
    :input: SMILES string (str)
    :return: graph object
    """

    mol = Chem.MolFromSmiles(smiles_string)
    if with_amap:
        max_amap = max([atom.GetAtomMapNum() for atom in mol.GetAtoms()])
        for atom in mol.GetAtoms():
            if atom.GetAtomMapNum() == 0:
                atom.SetAtomMapNum(max_amap + 1)
                max_amap = max_amap + 1

        amap_idx = {
            atom.GetAtomMapNum(): atom.GetIdx()
            for atom in mol.GetAtoms()
        }

    # atoms
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_features_list.append(atom_to_feature_vector(atom))
    x = np.array(atom_features_list, dtype=np.int64)

    # bonds
    num_bond_features = 3  # bond type, bond stereo, is_conjugated
    if len(mol.GetBonds()) > 0:  # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            edge_feature = bond_to_feature_vector(bond)

            # add edges in both directions
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = np.array(edges_list, dtype=np.int64).T

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = np.array(edge_features_list, dtype=np.int64)

    else:   # mol has no bonds
        edge_index = np.empty((2, 0), dtype=np.int64)
        edge_attr = np.empty((0, num_bond_features), dtype=np.int64)

    graph = dict()
    graph['edge_index'] = edge_index
    graph['edge_feat'] = edge_attr
    graph['node_feat'] = x
    graph['num_nodes'] = len(x)

    return graph if not with_amap else graph, amap_idx
