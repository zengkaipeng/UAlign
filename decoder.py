import torch
from sparse_backBone import SparseAtomEncoder, SparseBondEncoder
from Mix_backbone import MhAttnBlock


class Feat_init(torch.nn.Module):
	def __init__(
		self, n_pad, dim, heads=2, dropout=0.1, negative_slope=0.2
	):
		super(Feat_init, self).__init__()
		self.Kemb = torch.nn.Parameter(torch.randn(n_pad, dim))
		self.atom_encoder = SparseAtomEncoder(dim)
		self.bond_encoder = SparseBondEncoder(dim)
		self.Attn = MhAttnBlock(
			Qdim=dim, Kdim=dim, Vdim=dim, Odim=dim, 
			heads=heads, dropout=dropout
		)
		
