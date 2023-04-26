from model import edit_collect_fn
from utils.chemistry_parse import get_reaction_core, get_bond_info
from utils.graph_utils import smiles2graph


with open('utils/test_examples.txt') as Fin:
	content = Fin.readlines()



graphs, nodes, edge = [], [], []

for react in content:
	if len(react) <= 1:
		continue
	reac, prod = react.strip().split('>>')
	x, y, z = get_reaction_core(reac, prod)
	graph, amap = smiles2graph(prod)
	edge_types = get_bond_info()



