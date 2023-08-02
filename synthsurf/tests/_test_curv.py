import sys
import os
sys.path.insert(0, f'{os.path.dirname(__file__)}/../../')

from synthsurf.io import load_mesh
from synthsurf.surf import vertex_curv
from synthsurf.plotly import show_surf

SUB = 'sub-127989683000'
root = f'/autofs/space/pade_003/users/yb947/data/MRI/PSAMSEG/fs/{SUB}/surf'

v, f = load_mesh(f'{root}/lh.pial')
f = f.long()
v = v.float()

curv = vertex_curv(v, f, mode='max')

show_surf(v, f, intensity=curv, cmin=-1, cmax=1, colorscale='RdBu')

foo = 0