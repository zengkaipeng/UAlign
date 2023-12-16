from rdkit import Chem
from utils.chemistry_parse import BOND_FLOAT_TO_TYPE
from chem_test import qval_a_mole

qval_a_mole('[CH3:1][CH2:2][S:3][c:4]1[cH:5][cH:6][n:7][c:8](-[c:9]2[cH:10][n:11]3[c:12]([n:13][c:14]4[cH:15][cH:16][cH:17][cH:18][c:19]34)[s:20]2)[c:22]1[CH3:23].[OH:21][O:26][C:25](=[O:24])[c:27]1[cH:28][cH:29][cH:30][c:31]([Cl:32])[cH:33]1', '[CH3:1][CH2:2][S:3][c:4]1[cH:5][cH:6][n:7][c:8]([C:9]2=[CH:10][n:11]3[c:12]([n:13][c:14]4[cH:15][cH:16][cH:17][cH:18][c:19]34)[S:20]2=[O:21])[c:22]1[CH3:23]')