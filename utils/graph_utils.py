from ogb.utils.features import (
    allowable_features, atom_to_feature_vector, bond_feature_vector_to_dict,
    bond_to_feature_vector, atom_feature_vector_to_dict
)
import torch
import rdkit
from rdkit import Chem
import numpy as np


def smiles2graph(smiles_string, with_amap=False, kekulize=False):
    """
    Converts SMILES string to graph Data object
    :input: SMILES string (str)
    :return: graph object
    """

    mol = Chem.MolFromSmiles(smiles_string)
    if mol is not None and kekulize:
        Chem.Kekulize(mol)
    if with_amap:
        if len(mol.GetAtoms()) > 0:
            max_amap = max([atom.GetAtomMapNum() for atom in mol.GetAtoms()])
            for atom in mol.GetAtoms():
                if atom.GetAtomMapNum() == 0:
                    atom.SetAtomMapNum(max_amap + 1)
                    max_amap = max_amap + 1

            amap_idx = {
                atom.GetAtomMapNum(): atom.GetIdx()
                for atom in mol.GetAtoms()
            }
        else:
            amap_idx = dict()

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

    if with_amap:
        return graph, amap_idx
    else:
        return graph
